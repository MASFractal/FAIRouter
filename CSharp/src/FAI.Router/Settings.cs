using System;
using AI.DataStructs.Algebraic;
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

    /// <summary>
    /// Доля ходов, которые отдаются не лидеру, а случайному сопернику. Цена разведки состоит в
    /// том, что часть ходов заведомо достается не лучшему кандидату, зато журнал наполняется
    /// разнородными данными и обучению есть на чем учиться. Ноль отключает разведку целиком.
    /// </summary>
    public static double ExplorationRate { get; set; } = 0.2;

    /// <summary>
    /// Среднее по векторам задач. Вычитается из признаков перед подсчетом прогноза качества и
    /// перед обучением.
    /// </summary>
    /// <remarks>
    /// Векторы разных задач сонаправлены примерно на 0,95, поэтому у них велика общая
    /// составляющая. Она одинакова для всех кандидатов и полезных сведений не несет, зато
    /// смещения весов по ней в десятки раз сильнее смещений по различающей части, из-за чего
    /// кандидаты не расходятся по типам задач. Вычитание среднего убирает эту общую часть.
    /// <para>
    /// Значение задается один раз по накопленным ходам и дальше не меняется: веса обучены в
    /// пространстве с этим средним, и подмена среднего обесценивает их так же, как смена
    /// размерности признаков. Пустое значение отключает вычитание.
    /// </para>
    /// </remarks>
    public static Vector? TaskMean { get; set; }

    /// <summary>
    /// Приводит признаки задачи к тому виду, в котором работают прогноз качества и обучение:
    /// вычитает среднее и возвращает длину вектора к единице.
    /// </summary>
    /// <remarks>
    /// Возврат длины обязателен. Шаг обучения меняет оценку на величину, пропорциональную
    /// квадрату длины вектора задачи, а после вычитания среднего вектор укорачивается в разы.
    /// Обучение от этого замедляется во столько же раз, и разница между устройствами прогноза
    /// подменяется разницей скоростей. На измерении с тремя типами задач один только возврат
    /// длины поднял долю удачных запусков с 4 из 20 до 18 из 20.
    /// </remarks>
    /// <param name="features">Признаки задачи</param>
    public static Vector Center(Vector features)
    {
        if (TaskMean is null)
            return features;

        Vector centered = features - TaskMean;

        // Задача ровно в середине набора: направления у нее нет, и нормировать нечего
        return centered.NormL2() < 1e-12 ? centered : centered.GetUnitVector();
    }

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
    /// one-hot стиля (Style) + числовые метрики. Вычисляется, а не задается
    /// константой, чтобы не расходиться с Specifications.FeaturesSpecificationVector
    /// при добавлении новых стилей или метрик.
    /// </summary>
    public static int FeaturesSpecDim => Enum.GetValues<Style>().Length + SpecNumericFeaturesCount;
}
