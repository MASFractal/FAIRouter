using System.Text.RegularExpressions;
using AI.DataPrepaire.NLPUtils.RegexpNLP;
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
            HasReferences = Reference.IsMatch(text)
        };
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
