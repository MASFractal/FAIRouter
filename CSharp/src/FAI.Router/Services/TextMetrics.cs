using System.Text.RegularExpressions;
using FAI.Router.Enums;
using FAI.Router.JudgeLogic;

namespace FAI.Router.Services;

/// <summary>
/// Измерение спецификации текста по факту: все, что считается без обращения к LLM
/// </summary>
/// <remarks>
/// Правила общие с версией на Python и описаны здесь буквально, чтобы обе стороны мерили одинаково.
/// <list type="bullet">
/// <item>Слово: буквы и цифры, внутри соединенные дефисом, апострофом или подчеркиванием
/// (2025-м, gpt-4o, 3D-печать, snake_case это по одному слову, «15%» это слово 15). Так считают
/// текстовые редакторы; прежде C# и Python расходились на составных словах.</item>
/// <item>Код (ограды ``` и `встроенный`) в прозу не входит: структура, формулы, источники,
/// предложения, читаемость и язык меряются без него. Объем в символах и словах меряется по всему
/// тексту.</item>
/// <item>Предложение: строка прозы делится после . ! ? … перед пробелом и заглавной буквой
/// (A-Z, А-Я, Ё), цифрой, кавычкой или скобкой; конец строки тоже граница. Точки в сокращениях
/// (т.е., г., рис. и др. из <see cref="Abbreviation"/>, только с начала слова) границей не считаются.</item>
/// <item>Читаемость: индекс Флеша по прозе. Для русского коэффициенты Оборневой
/// 206,835 - 1,3 ASL - 60,1 ASW, где слог это гласная; для остальных Flesch Reading Ease
/// 206,835 - 1,015 ASL - 84,6 ASW, где слог это группа латинских гласных подряд, а немое окончание
/// (e, ed, es после согласной, кроме le, ted, ded и es после s, x, z, c, g) слогом не считается. Не меньше слога на слово.</item>
/// <item>Язык: ru или en по преобладающему алфавиту прозы; буквы вне русского алфавита у кириллицы
/// или с диакритикой у латиницы (больше 0,5 %), иная письменность или пустая проза значат
/// «неизвестно», а не en.</item>
/// <item>Формула: $$…$$, \[…\], \(…\) и $…$ только с признаками TeX внутри (\команда, ^, _, {):
/// «$5 и $10» формулой не считается.</item>
/// </list>
/// </remarks>
public static class TextMetrics
{
    /// <summary>Доля букв вне основного алфавита, после которой язык считается неизвестным</summary>
    private const double ForeignLetterShare = 0.005;

    private const char AbbreviationDot = '․';

    private static readonly Regex Word = new(@"[\p{L}\p{N}]+(?:['’\-_][\p{L}\p{N}]+)*", RegexOptions.Compiled);
    private static readonly Regex Heading = new(@"^(#{1,6})\s+\S", RegexOptions.Multiline | RegexOptions.Compiled);
    private static readonly Regex ListItem = new(@"^\s*([-*+]|\d+[.)])\s+\S", RegexOptions.Multiline | RegexOptions.Compiled);
    private static readonly Regex TableSeparator = new(@"^\s*\|[\s|:-]*-[\s|:-]*\|\s*$", RegexOptions.Multiline | RegexOptions.Compiled);
    private static readonly Regex CodeFence = new(@"^\s*```", RegexOptions.Multiline | RegexOptions.Compiled);
    private static readonly Regex CodeFenceInfo = new(@"^\s*```[ \t]*([A-Za-z0-9#+.\-]+)", RegexOptions.Multiline | RegexOptions.Compiled);
    private static readonly Regex CodeBlock = new(@"^[ \t]*```[^\n]*\n[\s\S]*?^[ \t]*```[ \t]*$", RegexOptions.Multiline | RegexOptions.Compiled);
    private static readonly Regex InlineCode = new(@"`[^`\n]+`", RegexOptions.Compiled);
    private static readonly Regex LinkTarget = new(@"!?\[([^\]]*)\]\([^)]*\)|https?://\S+", RegexOptions.Compiled);
    private static readonly Regex LineMarker = new(@"^\s*(#{1,6}\s+|[-*+>]\s+|\d+[.)]\s+)", RegexOptions.Multiline | RegexOptions.Compiled);
    private static readonly Regex SentenceEnd = new(@"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ0-9«""“(\[])", RegexOptions.Compiled);
    private static readonly Regex LatinVowels = new("[aeiouy]+", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex SilentEnding = new("[^aeiouyl]e$|[^aeiouytd]ed$|[^aeiouysxzcg]es$", RegexOptions.IgnoreCase | RegexOptions.Compiled);
    private static readonly Regex Formula = new(@"\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$(?=[^$\n]*[\\^_{])[^$\n]+\$", RegexOptions.Compiled);

    // Источник: ссылка разметки, адрес, DOI, строка списка литературы [1] … или сноска [^1]: …
    private static readonly Regex Reference = new(
        @"\[[^\]]+\]\([^)]+\)|https?://|\bdoi:\s*10\.\d{4,9}/|\b10\.\d{4,9}/\S+|^\s*\[\d{1,3}\]\s+\S|^\[\^[^\]]+\]:",
        RegexOptions.Multiline | RegexOptions.IgnoreCase | RegexOptions.Compiled);

    private static readonly Regex ParagraphBreak = new(@"\n\s*\n", RegexOptions.Compiled);

    /// <summary>
    /// Сокращения, после которых точка не заканчивает предложение. Совпадение только с начала слова:
    /// иначе «год.» прятал бы конец предложения как «д.».
    /// </summary>
    private static readonly Regex Abbreviation = new(
        @"(?<![\p{L}\p{N}])(?:и т\.д\.|и т\.п\.|т\.е\.|т\.д\.|т\.п\.|т\.к\.|т\.н\.|см\.|рис\.|табл\.|стр\.|гг\.|г\.|тыс\.|млн\.|млрд\.|им\.|ул\.|д\.|к\.|e\.g\.|i\.e\.|mr\.|mrs\.|ms\.|dr\.|vs\.)",
        RegexOptions.IgnoreCase | RegexOptions.Compiled);

    /// <summary>
    /// Подписи ограждений кода по языкам. Подпись вне таблицы означает язык вне списка.
    /// </summary>
    private static readonly Dictionary<string, ProgrammingLanguage> FenceLanguages = new(StringComparer.OrdinalIgnoreCase)
    {
        ["python"] = ProgrammingLanguage.Python, ["py"] = ProgrammingLanguage.Python,
        ["javascript"] = ProgrammingLanguage.JavaScript, ["js"] = ProgrammingLanguage.JavaScript,
        ["jsx"] = ProgrammingLanguage.JavaScript, ["node"] = ProgrammingLanguage.JavaScript,
        ["typescript"] = ProgrammingLanguage.TypeScript, ["ts"] = ProgrammingLanguage.TypeScript,
        ["tsx"] = ProgrammingLanguage.TypeScript,
        ["csharp"] = ProgrammingLanguage.CSharp, ["cs"] = ProgrammingLanguage.CSharp, ["c#"] = ProgrammingLanguage.CSharp,
        ["java"] = ProgrammingLanguage.Java,
        ["go"] = ProgrammingLanguage.Go, ["golang"] = ProgrammingLanguage.Go,
        ["rust"] = ProgrammingLanguage.Rust, ["rs"] = ProgrammingLanguage.Rust,
        ["cpp"] = ProgrammingLanguage.Cpp, ["c++"] = ProgrammingLanguage.Cpp, ["cc"] = ProgrammingLanguage.Cpp,
        ["cxx"] = ProgrammingLanguage.Cpp, ["c"] = ProgrammingLanguage.Cpp, ["h"] = ProgrammingLanguage.Cpp,
        ["sql"] = ProgrammingLanguage.Sql, ["postgresql"] = ProgrammingLanguage.Sql,
        ["mysql"] = ProgrammingLanguage.Sql, ["psql"] = ProgrammingLanguage.Sql,
        ["html"] = ProgrammingLanguage.Html, ["xml"] = ProgrammingLanguage.Html,
        ["css"] = ProgrammingLanguage.Html, ["svg"] = ProgrammingLanguage.Html,
    };

    /// <summary>
    /// Измеряет спецификацию готового текста.
    /// StyleType, TermDensity и FormalityScore остаются по умолчанию, так как
    /// они смысловые и измеряются моделью.
    /// </summary>
    /// <param name="text">Текст ответа</param>
    public static Specifications Measure(string text)
    {
        string body = WithoutCode(text);
        string prose = Prose(body);
        string? language = DetectLanguage(prose);
        string[] proseWords = [.. Word.Matches(prose).Select(match => match.Value)];
        int sentences = CountSentences(prose);
        int[] headingLevels = [.. Heading.Matches(body).Select(match => match.Groups[1].Value.Length)];

        return new Specifications
        {
            SymbolLength = text.Length,
            WordLength = CountWords(text),
            ParagraphCount = CountParagraphs(body),
            SectionCount = headingLevels.Length == 0 ? 0 : headingLevels.Count(level => level == headingLevels.Min()),
            ListItemCount = ListItem.Matches(body).Count,
            TableCount = TableSeparator.Matches(body).Count,
            CodeBlockCount = CodeFence.Matches(text).Count / 2,
            FormulaCount = Formula.Matches(body).Count,
            HeadingDepth = headingLevels.Length == 0 ? 0 : headingLevels.Max(),
            AvgSentenceLength = sentences == 0 ? 0 : (double)proseWords.Length / sentences,
            ReadabilityScore = Readability(proseWords, sentences, language),
            Language = language,
            HasReferences = Reference.IsMatch(body),
            ProgrammingLanguage = ProgrammingLanguageOf(text)
        };
    }

    /// <summary>
    /// Язык кода в тексте по подписям ограждений: самый частый из подписанных, при равенстве тот,
    /// что встретился раньше. Подпись вне таблицы дает язык вне списка (Other). Блоков нет, тогда None.
    /// Блоки есть, а подписей нет, тогда язык неизвестен и это тоже None (вместе с ненулевым числом
    /// блоков): прежде это был Other, и заказанный Python с неподписанным блоком считался расхождением.
    /// </summary>
    /// <param name="text">Текст ответа</param>
    public static ProgrammingLanguage ProgrammingLanguageOf(string text) =>
        CodeFenceInfo.Matches(text)
            .Select(match => FenceLanguages.GetValueOrDefault(match.Groups[1].Value, ProgrammingLanguage.Other))
            .GroupBy(language => language)
            .OrderByDescending(group => group.Count())
            .Select(group => group.Key)
            .FirstOrDefault(ProgrammingLanguage.None);

    /// <summary>Число слов по общему правилу (см. описание класса)</summary>
    /// <param name="text">Текст</param>
    public static int CountWords(string text) => Word.Matches(text).Count;

    /// <summary>Число предложений по общему правилу (см. описание класса)</summary>
    /// <param name="text">Проза без кода</param>
    public static int CountSentences(string text) =>
        text.Split('\n')
            .Select(line => Abbreviation.Replace(line, match => match.Value.Replace('.', AbbreviationDot)))
            .Sum(line => SentenceEnd.Split(line).Count(part => part.Any(char.IsLetterOrDigit)));

    /// <summary>
    /// Язык прозы кодом ISO 639-1: ru или en; пусто, если язык иной или не определяется
    /// </summary>
    /// <param name="prose">Проза без кода</param>
    public static string? DetectLanguage(string prose)
    {
        int letters = 0, cyrillic = 0, russian = 0, latin = 0, ascii = 0;

        foreach (char symbol in prose.Where(char.IsLetter))
        {
            letters++;

            if (symbol is >= 'Ѐ' and <= 'ӿ')
            {
                cyrillic++;
                russian += symbol is >= 'а' and <= 'я' or >= 'А' and <= 'Я' or 'ё' or 'Ё' ? 1 : 0;
            }
            else if (symbol is >= 'a' and <= 'z' or >= 'A' and <= 'Z' or >= 'À' and <= 'ɏ')
            {
                latin++;
                ascii += symbol < '\u0080' ? 1 : 0;
            }
        }

        if (cyrillic >= latin && cyrillic * 2 > letters)
            return cyrillic - russian > ForeignLetterShare * cyrillic ? null : "ru";

        if (latin > cyrillic && latin * 2 > letters)
            return latin - ascii > ForeignLetterShare * latin ? null : "en";

        return null;
    }

    // Текст без блоков кода: блок заменяется пустой строкой, и соседние абзацы не склеиваются
    private static string WithoutCode(string text) => CodeBlock.Replace(text, "\n");

    // Проза: без встроенного кода, адресов и разметки ссылок (текст ссылки остается), без меток
    // заголовков, списков и цитат, без строк-разделителей таблиц; ячейки таблиц идут отдельными строками
    private static string Prose(string body)
    {
        string text = TableSeparator.Replace(body, "");
        text = InlineCode.Replace(text, " ");
        text = LinkTarget.Replace(text, match => match.Groups[1].Value);
        text = LineMarker.Replace(text, "");

        return text.Replace('|', '\n').Replace("*", "");
    }

    // Абзацы разделены пустой строкой
    private static int CountParagraphs(string text) =>
        ParagraphBreak.Split(text).Count(IsProse);

    // Абзацем считается сплошной текст: заголовок, список, таблица и код абзацами не считаются,
    // иначе замер разойдется с заказом, где под абзацами понимают прозу
    private static bool IsProse(string block)
    {
        string? firstLine = block.Split('\n').FirstOrDefault(line => !string.IsNullOrWhiteSpace(line));

        if (firstLine == null)
            return false;

        return !Heading.IsMatch(firstLine)
            && !ListItem.IsMatch(firstLine)
            && !CodeFence.IsMatch(firstLine)
            && !firstLine.TrimStart().StartsWith('|');
    }

    // Индекс удобочитаемости Флеша, 0-100: коэффициенты Оборневой для русского, Reading Ease для прочих
    private static double Readability(string[] words, int sentences, string? language)
    {
        if (words.Length == 0 || sentences == 0)
            return 0;

        double asl = (double)words.Length / sentences;
        double asw = words.Sum(Syllables) / (double)words.Length;
        double score = language == "ru"
            ? 206.835 - 1.3 * asl - 60.1 * asw
            : 206.835 - 1.015 * asl - 84.6 * asw;

        return Math.Clamp(score, 0, 100);
    }

    // Слоги слова: русские гласные и группы латинских гласных подряд без немого окончания, не меньше одного
    private static int Syllables(string word)
    {
        int latin = LatinVowels.Matches(word).Count;

        if (latin > 1 && SilentEnding.IsMatch(word))
            latin--;

        return Math.Max(1, word.Count(symbol => "аеёиоуыэюя".Contains(char.ToLowerInvariant(symbol))) + latin);
    }
}
