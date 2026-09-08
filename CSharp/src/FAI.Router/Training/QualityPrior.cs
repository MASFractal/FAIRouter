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
    /// <summary>
    /// Строит вектор кандидата так, чтобы прогноз качества на замеренных задачах совпал с замером
    /// </summary>
    /// <remarks>
    /// Замеров всегда меньше, чем координат вектора, поэтому решений бесконечно много и берется
    /// самое короткое. Такой вектор лежит в линейной оболочке замеренных задач, то есть равен
    /// сумме их векторов с коэффициентами, а коэффициенты находятся из системы Грама.
    /// Добавка к диагонали удерживает решение, когда замеренные задачи почти сонаправлены.
    /// </remarks>
    /// <param name="measurements">Пары «признаки задачи и замеренное качество кандидата на ней»</param>
    /// <param name="ridge">Добавка к диагонали системы</param>
    public static Vector FromMeasurements(IReadOnlyList<(Vector Task, double Quality)> measurements, double ridge = 1e-3)
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
                gram[row][column] = tasks[row].Dot(tasks[column]) + (row == column ? ridge : 0);
        }

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
