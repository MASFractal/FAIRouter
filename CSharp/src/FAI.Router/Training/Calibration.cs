namespace FAI.Router.Training;

/// <summary>
/// Калибровка прогноза качества в вероятность того, что ответ устроит человека: p = σ(A·q + B).
/// </summary>
/// <remarks>
/// Прогноз кандидата относительный: скалярное произведение признаков задачи на его вектор умеет
/// сказать «этот лучше того», но не «этот сойдет». Планке достаточности нужна абсолютная шкала, и
/// калибровка переводит прогноз в долю лайков, которую такой прогноз получал на деле.
/// <para>
/// Параметра два, поэтому решатель свой: метод Ньютона на системе 2×2. Готовая логистическая
/// регрессия в AIFramework (AI.Econometrics) внутренняя и лежит в сборке, на которую библиотека не
/// ссылается.
/// </para>
/// </remarks>
/// <param name="A">Наклон: насколько прогноз вообще говорит о качестве</param>
/// <param name="B">Сдвиг</param>
public readonly record struct Calibration(double A, double B)
{
    /// <summary>Сколько шагов Ньютона разрешено: на двух параметрах сходится за десяток.</summary>
    private const int MaxIterations = 50;

    /// <summary>Шаг, меньше которого решение считается найденным.</summary>
    private const double Tolerance = 1e-10;

    /// <summary>
    /// Слабая привязка сдвига к доле лайков. Когда все оценки одинаковые, у сдвига нет конечного
    /// оптимума, и без привязки он уходил бы в бесконечность.
    /// </summary>
    private const double ShiftRidge = 1e-3;

    /// <summary>Вероятность, что ответ с таким прогнозом устроит человека.</summary>
    /// <param name="quality">Прогноз качества кандидата</param>
    public double Predict(double quality) => Sigmoid(A * quality + B);

    /// <summary>
    /// Подбирает калибровку по парам «прогноз и оценка».
    /// </summary>
    /// <remarks>
    /// Наклон стягивается к нулю (<paramref name="ridge"/>): пока оценок мало, прогноз считается
    /// малоинформативным, и калибровка отдает почти одну долю лайков, а не выдумывает зависимость.
    /// </remarks>
    /// <param name="pairs">Прогноз в момент выбора и оценка от 0 до 1</param>
    /// <param name="ridge">Сила стягивания наклона к нулю</param>
    public static Calibration Fit(IReadOnlyList<(double Quality, double Score)> pairs, double ridge = 1.0)
    {
        if (pairs.Count == 0)
            throw new ArgumentException("Нужна хотя бы одна оценка.", nameof(pairs));

        double rate = Math.Clamp(pairs.Average(pair => pair.Score), 1e-3, 1 - 1e-3);
        double anchor = Math.Log(rate / (1 - rate));
        double a = 0, b = anchor;

        for (int step = 0; step < MaxIterations; step++)
        {
            double gA = ridge * a, gB = ShiftRidge * (b - anchor);
            double hAA = ridge, hAB = 0, hBB = ShiftRidge;

            foreach ((double q, double y) in pairs)
            {
                double p = Sigmoid(a * q + b);
                double w = p * (1 - p);
                gA += (p - y) * q;
                gB += p - y;
                hAA += w * q * q;
                hAB += w * q;
                hBB += w;
            }

            double det = hAA * hBB - hAB * hAB;
            if (det < 1e-18)
                break;

            double dA = (hBB * gA - hAB * gB) / det;
            double dB = (hAA * gB - hAB * gA) / det;
            a -= dA;
            b -= dB;

            if (Math.Abs(dA) < Tolerance && Math.Abs(dB) < Tolerance)
                break;
        }

        return new Calibration(a, b);
    }

    // Показатель ограничен: на краях экспонента иначе дает бесконечность и NaN в производной
    private static double Sigmoid(double x) => 1 / (1 + Math.Exp(-Math.Clamp(x, -35, 35)));
}
