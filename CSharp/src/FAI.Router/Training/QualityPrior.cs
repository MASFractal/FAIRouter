using AI.DataStructs.Algebraic;

namespace FAI.Router.Training;

/// <summary>
/// Начальные веса кандидата по заранее замеренному качеству на задачах известных типов.
/// </summary>
/// <remarks>
/// Без этого вектор кандидата задается по Ксавье, то есть случайно, и первый выбор роутера
/// оказывается жребием. Замер по типам задач стоит нескольких десятков обращений к моделям и
/// делается один раз, зато система приносит пользу с первого хода, а обучение дальше только
/// уточняет то, что уже известно.
/// </remarks>
public static class QualityPrior
{
    /// <summary>Добавка к диагонали по умолчанию, доля среднего квадрата длины задачи (см. <see cref="FromMeasurements"/>)</summary>
    public const double DefaultRidge = 0.3;

    /// <summary>
    /// Строит вектор кандидата так, чтобы прогноз качества на замеренных задачах совпал с замером
    /// </summary>
    /// <remarks>
    /// <para>
    /// Замеров всегда меньше, чем координат вектора, поэтому решений бесконечно много и берется
    /// самое короткое. Такой вектор лежит в линейной оболочке замеренных задач, то есть равен
    /// сумме их векторов с коэффициентами, а коэффициенты находятся из системы Грама.
    /// </para>
    /// <para>
    /// Добавка к диагонали — ДОЛЯ среднего квадрата длины задачи, а не абсолютное число: признаки
    /// приходят и нормированными (длина 1), и сырыми, где длина у каждой модели своя, и одно
    /// абсолютное число значило бы в них разное. Прежняя добавка 1e-3 почти точно проводила
    /// прогноз через все точки рейтингов, а их у модели полсотни, близких и с разными оценками:
    /// коэффициенты раздувались, длина вектора доходила до 10, и на реальных запросах, лежащих в
    /// стороне от точек рейтингов, прогноз уходил к 3 при шкале до 1. Выигрывала модель с самым
    /// раздутым вектором, а не с лучшими оценками, и один и тот же кандидат побеждал на любой задаче.
    /// Доля 0,3 выбрана по ошибке «выбрось точку и предскажи ее» на рейтингах 263 моделей
    /// каталога (замер 21.09.2026, снимок рейтингов от 20.09): 0,241 при 1e-3, 0,182 при 0,3, 0,183 при 1; прогнозов вне
    /// [−0,2; 1,2] на реальных запросах 28 % при 1e-3 и ни одного начиная с 0,01.
    /// </para>
    /// </remarks>
    /// <param name="measurements">Пары «признаки задачи и замеренное качество кандидата на ней»</param>
    /// <param name="ridge">Добавка к диагонали системы, доля среднего квадрата длины задачи</param>
    public static Vector FromMeasurements(IReadOnlyList<(Vector Task, double Quality)> measurements, double ridge = DefaultRidge)
    {
        if (measurements.Count == 0)
            throw new ArgumentException("Нужен хотя бы один замер.", nameof(measurements));

        // Тот же вид признаков, в котором работают прогноз и обучение
        Vector[] tasks = [.. measurements.Select(item => Settings.Center(item.Task))];
        int count = tasks.Length;

        double[][] gram = new double[count][];

        for (int row = 0; row < count; row++)
        {
            gram[row] = new double[count];

            for (int column = 0; column < count; column++)
                gram[row][column] = tasks[row].Dot(tasks[column]);
        }

        double scale = ridge * Enumerable.Range(0, count).Average(i => gram[i][i]);

        for (int row = 0; row < count; row++)
            gram[row][row] += scale;

        double[] coefficients = Solve(gram, [.. measurements.Select(item => item.Quality)]);
        Vector prior = new(tasks[0].Count);

        for (int i = 0; i < count; i++)
            prior += tasks[i] * coefficients[i];

        return prior;
    }

    // Решение малой системы исключением Гаусса с выбором главного элемента.
    // Искал Inverse, Solve и Gauss в AI и AI.ClassicMath, готового решателя не нашлось,
    // а система здесь размером с число замеренных типов задач, то есть единицы.
    private static double[] Solve(double[][] matrix, double[] right)
    {
        int size = right.Length;

        for (int step = 0; step < size; step++)
        {
            int pivot = step;

            for (int row = step + 1; row < size; row++)
                if (Math.Abs(matrix[row][step]) > Math.Abs(matrix[pivot][step]))
                    pivot = row;

            (matrix[step], matrix[pivot]) = (matrix[pivot], matrix[step]);
            (right[step], right[pivot]) = (right[pivot], right[step]);

            // Замеры повторяют друг друга: строка вырождена, и вклад этого замера обнуляется
            if (Math.Abs(matrix[step][step]) < 1e-12)
                continue;

            for (int row = step + 1; row < size; row++)
            {
                double factor = matrix[row][step] / matrix[step][step];

                for (int column = step; column < size; column++)
                    matrix[row][column] -= factor * matrix[step][column];

                right[row] -= factor * right[step];
            }
        }

        double[] solution = new double[size];

        for (int row = size - 1; row >= 0; row--)
        {
            if (Math.Abs(matrix[row][row]) < 1e-12)
                continue;

            double sum = right[row];

            for (int column = row + 1; column < size; column++)
                sum -= matrix[row][column] * solution[column];

            solution[row] = sum / matrix[row][row];
        }

        return solution;
    }
}
