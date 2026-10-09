using FAI.Router.Enums;
using FAI.Router.JudgeLogic;
using FAI.Router.Services;

namespace FAI.Router.Tests;

/// <summary>Замер текста и сверка заказа с фактом по пунктам</summary>
public class MeasureTests
{
    /// <summary>Составные слова считаются одним словом, как в текстовых редакторах</summary>
    [Fact]
    public void Compound_words_count_once()
    {
        Assert.Equal(10, TextMetrics.CountWords("Рост на 15% в 2025-м году, модель gpt-4o и 3D-печать."));
        Assert.Equal(1, TextMetrics.CountWords("snake_case"));
    }

    /// <summary>Точка в сокращении не конец предложения, а конец слова на «д» конец</summary>
    [Fact]
    public void Sentences_respect_abbreviations_only_at_word_start()
    {
        Assert.Equal(2, TextMetrics.CountSentences("Прошел год. Новый начался."));
        Assert.Equal(1, TextMetrics.CountSentences("Это видно на рис. 3 и в табл. 2."));
        Assert.Equal(2, TextMetrics.CountSentences("Заголовок без точки\nстрока списка"));
    }

    /// <summary>Доллары с числами формулой не считаются, TeX считается</summary>
    [Fact]
    public void Formulas_need_tex_markup()
    {
        Assert.Equal(0, TextMetrics.Measure("Цена $5 и $10 за штуку.").FormulaCount);
        Assert.Equal(3, TextMetrics.Measure("Пусть $x^2$ и $$a+b$$, а также \\(\\alpha\\).").FormulaCount);
    }

    /// <summary>Комментарий в коде не заголовок, код не входит в язык и читаемость</summary>
    [Fact]
    public void Code_is_not_prose()
    {
        Specifications spec = TextMetrics.Measure("Пример ниже.\n\n```python\n# comment\nprint('hello world')\n```\n");

        Assert.Equal(0, spec.SectionCount);
        Assert.Equal("ru", spec.Language);
        Assert.Equal(1, spec.CodeBlockCount);
        Assert.Equal(ProgrammingLanguage.Python, spec.ProgrammingLanguage);
    }

    /// <summary>Язык вне ru и en честно неизвестен, а не en</summary>
    [Fact]
    public void Unknown_language_is_not_english()
    {
        Assert.Equal("en", TextMetrics.DetectLanguage("The quick brown fox jumps over the lazy dog."));
        Assert.Null(TextMetrics.DetectLanguage("Le café est très agréable près de la gare, déjà ouvert à midi."));
        Assert.Null(TextMetrics.DetectLanguage("Він прийшов і її побачив."));
        Assert.Null(TextMetrics.DetectLanguage("你好，世界"));
    }

    /// <summary>Читаемость английского по Flesch Reading Ease: у этого текста по словарным слогам около 44, прежняя формула давала под 100</summary>
    [Fact]
    public void English_readability_uses_reading_ease()
    {
        const string text = "The committee reviewed the proposal carefully before the meeting. "
            + "Several members raised concerns about the budget and the timeline. "
            + "After a long discussion, they agreed to postpone the final decision until next month.";

        double score = TextMetrics.Measure(text).ReadabilityScore;

        Assert.InRange(score, 34, 54);
    }

    /// <summary>Неподписанный блок кода не дает языка, и заказанный Python с ним не расходится</summary>
    [Fact]
    public void Unlabeled_code_is_not_a_language_mismatch()
    {
        Specifications actual = TextMetrics.Measure("Код:\n\n```\nprint(1)\n```\n");
        Specifications requested = new() { ProgrammingLanguage = ProgrammingLanguage.Python, CodeBlockCount = 1, ExplicitFields = ["programmingLanguage"] };

        Assert.Equal(ProgrammingLanguage.None, actual.ProgrammingLanguage);
        Assert.DoesNotContain(DiffSpec.Compare(requested, actual).Deviations, item => item.Field == "Язык программирования");
    }

    /// <summary>Шкала с границами: ответ экспертнее заказа штрафуется не сильнее, чем такой же недобор</summary>
    [Fact]
    public void Bounded_scales_use_absolute_difference()
    {
        Specifications requested = new() { FormalityScore = 0.2 };

        double over = Deviation(DiffSpec.Compare(requested, new Specifications { FormalityScore = 0.9 }), "Формальность");
        double under = Deviation(DiffSpec.Compare(new Specifications { FormalityScore = 0.9 }, new Specifications { FormalityScore = 0.2 }), "Формальность");

        Assert.Equal(0.7, over, 9);
        Assert.Equal(over, under, 9);
    }

    /// <summary>Угаданные поля не сверяются, объем одной строкой, незаказанные источники не штрафуются</summary>
    [Fact]
    public void Only_stated_fields_are_checked_strictly()
    {
        Specifications requested = new()
        {
            SymbolLength = 3000, WordLength = 450, TableCount = 0, Language = "RU",
            ExplicitFields = ["symbolLength", "wordLength", "language"],
        };
        Specifications actual = new() { SymbolLength = 3000, WordLength = 900, TableCount = 2, Language = "ru", HasReferences = true };

        DiffSpec critic = DiffSpec.Compare(requested, actual);
        string[] fields = [.. critic.Deviations.Select(item => item.Field)];

        Assert.Contains("Объем в символах", fields);
        Assert.DoesNotContain("Объем в словах", fields);
        Assert.DoesNotContain("Таблицы", fields);
        Assert.DoesNotContain("Ссылки на источники", fields);
        Assert.Equal(0, Deviation(critic, "Язык"));
        Assert.Equal(0, critic.FormDeviation, 9);
    }

    /// <summary>Заказ без списка явных полей сверяется целиком, но незаказанные источники не штрафуются</summary>
    [Fact]
    public void Legacy_order_checks_every_field_without_punishing_sources()
    {
        Specifications requested = new() { Language = "ru-RU" };
        Specifications actual = new() { Language = "ru", HasReferences = true };

        DiffSpec critic = DiffSpec.Compare(requested, actual);

        Assert.Equal(0, Deviation(critic, "Ссылки на источники"));
        Assert.Equal(0, Deviation(critic, "Язык"));
        Assert.Contains(critic.Deviations, item => item.Field == "Таблицы");
        Assert.Equal("Таблицы: заказано 0, получено 0", critic.Deviations.First(item => item.Field == "Таблицы").ToString());
    }

    private static double Deviation(DiffSpec critic, string field) => critic.Deviations.Single(item => item.Field == field).Deviation;
}
