using System;
using System.Collections.Generic;
using System.Text;
using AI.LLM.Services.LLM;
using FAI.Router.Enums;

namespace FAI.Router;

public class Settings
{
    public static double WQ { get; set; } = 0.5;
    public static double Wt { get; set; } = 0.25;
    public static double WC { get; set; } = 0.25;

    private static LLMBase? _llm;

    /// <summary>
    /// Модель для работы системы. Назначается один раз при старте приложения.
    /// </summary>
    public static LLMBase LLM
    {
        get => _llm ?? throw new InvalidOperationException(
            "Settings.LLM не задан: назначьте клиент LLM до обращения к сервисам распознавания.");
        set => _llm = value;
    }

    /// <summary>
    /// Размерность пространства признаков
    /// </summary>
    public static int FeaturesDim { get; set; } = 3;


    /// <summary>
    /// Число числовых метрик в Specifications (кроме one-hot стиля):
    /// SymbolLength, WordLength, ParagraphCount, SectionCount, ListItemCount,
    /// TableCount, CodeBlockCount, FormulaCount, HeadingDepth, AvgSentenceLength,
    /// ReadabilityScore, TermDensity, FormalityScore
    /// </summary>
    private const int SpecNumericFeaturesCount = 13;

    /// <summary>
    /// Размерность пространства признаков спецификации:
    /// one-hot стиля (Style) + числовые метрики. Вычисляется, а не задаётся
    /// константой, чтобы не расходиться с Specifications.FeaturesSpecificationVector
    /// при добавлении новых стилей или метрик.
    /// </summary>
    public static int FeaturesSpecDim => Enum.GetValues<Style>().Length + SpecNumericFeaturesCount;
}
