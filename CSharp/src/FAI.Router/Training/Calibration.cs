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

    /// <summary>Наибольшая длина шага Ньютона: прогноз и сдвиг живут на шкале единиц</summary>
    private const double MaxStep = 10;

    /// <summary>Сколько раз шаг можно уменьшить вдвое, прежде чем принять его как есть</summary>
    private const int MaxHalvings = 30;

    /// <summary>
    /// Слабая привязка сдвига к доле лайков. Когда все оценки одинаковые, у сдвига нет конечного
    /// оптимума, и без привязки он уходил бы в бесконечность.
    /// </summary>
    private const double ShiftRidge = 1e-3;

    /// <summary>Вероятность, что ответ с таким прогнозом устроит человека.</summary>
    /// <param name="quality">Прогноз качества кандидата</param>
    public double Predict(double quality) => Sigmoid(A * quality + B);

    /// <summary>
    /// Сила стягивания наклона к нулю по умолчанию, см. <see cref="Fit"/>
    /// </summary>
    public const double DefaultRidge = 0.1;

    /// <summary>
    /// Подбирает калибровку по парам «прогноз и оценка».
    /// </summary>
    /// <remarks>
    /// Наклон стягивается к нулю (<paramref name="ridge"/>): пока оценок мало, прогноз считается
    /// малоинформативным, и калибровка отдает почти одну долю лайков, а не выдумывает зависимость.
    /// <para>
    /// Сила стягивания по умолчанию 0,1. До 05.10.2026 она равнялась 1, и наклон оставался заниженным
    /// даже на сотнях оценок: 5,3 при истинном 8 на 500 оценках. Значение выбрано по избыточной
    /// логистической ошибке на отложенных точках (наклоны 3, 8 и 16, от 3 до 500 оценок, по 300
    /// выборок): в среднем 0,337 при 1, 0,241 при 0,1, 0,225 при 0,01. Слабее 0,1 стягивать не стали:
    /// на 3-10 оценках и пологой зависимости ошибка тогда растет.
    /// </para>
    /// </remarks>
    /// <param name="pairs">Прогноз в момент выбора и оценка от 0 до 1</param>
    /// <param name="ridge">Сила стягивания наклона к нулю</param>
    public static Calibration Fit(IReadOnlyList<(double Quality, double Score)> pairs, double ridge = DefaultRidge)
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
            if (!(det >= 1e-18))
                break;

            double dA = (hBB * gA - hAB * gB) / det;
            double dB = (hAA * gB - hAB * gA) / det;

            if (!double.IsFinite(dA) || !double.IsFinite(dB))
                break;

            // Шаг ограничен и уменьшается вдвое, пока не снизит цель: на выбросах и почти
            // разделимых выборках полный шаг Ньютона перескакивал минимум и уходил в бесконечность
            double norm = Math.Sqrt(dA * dA + dB * dB);
            double scale = norm > MaxStep ? MaxStep / norm : 1;
            double current = Objective(pairs, a, b, ridge, anchor);

            for (int halving = 0; halving < MaxHalvings && Objective(pairs, a - scale * dA, b - scale * dB, ridge, anchor) > current + 1e-12; halving++)
                scale /= 2;

            a -= scale * dA;
            b -= scale * dB;

            if (Math.Abs(scale * dA) < Tolerance && Math.Abs(scale * dB) < Tolerance)
                break;
        }

        return double.IsFinite(a) && double.IsFinite(b) ? new Calibration(a, b) : new Calibration(0, anchor);
    }

    // Цель подгонки: логистическая ошибка плюс стягивание наклона к нулю и сдвига к доле лайков
    private static double Objective(IReadOnlyList<(double Quality, double Score)> pairs, double a, double b, double ridge, double anchor)
    {
        double sum = 0.5 * ridge * a * a + 0.5 * ShiftRidge * (b - anchor) * (b - anchor);

        foreach ((double q, double y) in pairs)
        {
            double z = Math.Clamp(a * q + b, -35, 35);

            // log(1 + e^z) без переполнения
            sum += Math.Max(z, 0) + Math.Log(1 + Math.Exp(-Math.Abs(z))) - y * z;
        }

        return sum;
    }

    // Показатель ограничен: на краях экспонента иначе дает бесконечность и NaN в производной
    private static double Sigmoid(double x) => 1 / (1 + Math.Exp(-Math.Clamp(x, -35, 35)));
}
