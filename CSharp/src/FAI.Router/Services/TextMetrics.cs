using System.Text.RegularExpressions;
using AI.DataPrepaire.NLPUtils.RegexpNLP;
using FAI.Router.Enums;
using FAI.Router.JudgeLogic;

namespace FAI.Router.Services;

/// <summary>
/// Измерение спецификации текста по факту: все, что считается без обращения к LLM
/// </summary>
public static class TextMetrics
{
    /// <summary>
    /// Гласные русского и английского алфавитов (слог ≈ гласная)
    /// </summary>
    private const string Vowels = "аеёиоуыэюяaeiouy";

    private static readonly SentencesTokenizer Sentences = new();

    private static readonly Regex Word = new(@"[\p{L}\p{N}][\p{L}\p{N}\-']*", RegexOptions.Compiled);
    private static readonly Regex Heading = new(@"^(#{1,6})\s+\S", RegexOptions.Multiline | RegexOptions.Compiled);
    private static readonly Regex ListItem = new(@"^\s*([-*+]|\d+[.)])\s+\S", RegexOptions.Multiline | RegexOptions.Compiled);
    private static readonly Regex TableSeparator = new(@"^\s*\|[\s|:-]*-[\s|:-]*\|\s*$", RegexOptions.Multiline | RegexOptions.Compiled);
    private static readonly Regex CodeFence = new(@"^\s*```", RegexOptions.Multiline | RegexOptions.Compiled);
    private static readonly Regex CodeFenceInfo = new(@"^\s*```[ \t]*([A-Za-z0-9#+.\-]+)", RegexOptions.Multiline | RegexOptions.Compiled);

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
    private static readonly Regex Formula = new(@"\$\$[^$]+\$\$|\$[^$\n]+\$", RegexOptions.Compiled);
    private static readonly Regex Reference = new(@"\[[^\]]+\]\([^)]+\)|https?://", RegexOptions.Compiled);
    private static readonly Regex ParagraphBreak = new(@"\n\s*\n", RegexOptions.Compiled);

    /// <summary>
    /// Измеряет спецификацию готового текста.
    /// StyleType, TermDensity и FormalityScore остаются по умолчанию, так как
    /// они смысловые и измеряются моделью.
    /// </summary>
    /// <param name="text">Текст ответа</param>
    public static Specifications Measure(string text)
    {
        int words = Word.Matches(text).Count;
        int sentences = Sentences.Tokenize(text).Count;
        int[] headingLevels = [.. Heading.Matches(text).Select(match => match.Groups[1].Value.Length)];

        return new Specifications
        {
            SymbolLength = text.Length,
            WordLength = words,
            ParagraphCount = CountParagraphs(text),
            SectionCount = headingLevels.Length == 0 ? 0 : headingLevels.Count(level => level == headingLevels.Min()),
            ListItemCount = ListItem.Matches(text).Count,
            TableCount = TableSeparator.Matches(text).Count,
            CodeBlockCount = CodeFence.Matches(text).Count / 2,
            FormulaCount = Formula.Matches(text).Count,
            HeadingDepth = headingLevels.Length == 0 ? 0 : headingLevels.Max(),
            AvgSentenceLength = sentences == 0 ? 0 : (double)words / sentences,
            ReadabilityScore = Readability(text, words, sentences),
            Language = DetectLanguage(text),
            HasReferences = Reference.IsMatch(text),
            ProgrammingLanguage = ProgrammingLanguageOf(text)
        };
    }

    /// <summary>
    /// Язык кода в тексте по подписям ограждений: самый частый из подписанных. Блоки есть, а
    /// подписей нет, тогда язык вне списка; блоков нет, тогда кода не написано.
    /// </summary>
    /// <param name="text">Текст ответа</param>
    public static ProgrammingLanguage ProgrammingLanguageOf(string text)
    {
        var tagged = CodeFenceInfo.Matches(text)
            .Select(match => FenceLanguages.GetValueOrDefault(match.Groups[1].Value, ProgrammingLanguage.Other))
            .GroupBy(language => language)
            .OrderByDescending(group => group.Count())
            .Select(group => group.Key)
            .ToList();

        if (tagged.Count > 0)
            return tagged[0];

        return CodeFence.IsMatch(text) ? ProgrammingLanguage.Other : ProgrammingLanguage.None;
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

    // Индекс удобочитаемости Флеша в адаптации Оборневой для русского языка, 0-100
    private static double Readability(string text, int words, int sentences)
    {
        if (words == 0 || sentences == 0)
            return 0;

        double syllables = text.Count(symbol => Vowels.Contains(char.ToLowerInvariant(symbol)));
        double score = 206.835 - 1.3 * words / sentences - 60.1 * syllables / words;

        return Math.Clamp(score, 0, 100);
    }

    // Язык по преобладанию алфавита: различает русский и английский
    private static string? DetectLanguage(string text)
    {
        int cyrillic = text.Count(symbol => symbol is >= 'а' and <= 'я' or >= 'А' and <= 'Я' or 'ё' or 'Ё');
        int latin = text.Count(symbol => symbol is >= 'a' and <= 'z' or >= 'A' and <= 'Z');

        if (cyrillic == 0 && latin == 0)
            return null;

        return cyrillic >= latin ? "ru" : "en";
    }
}
