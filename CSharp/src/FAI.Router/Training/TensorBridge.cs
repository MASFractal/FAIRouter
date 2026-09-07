using AI.DataStructs.Algebraic;
using AI.ML.NeuralNetworks.V2;

// В AIFramework два разных Tensor: алгебраический и тензор автограда. Здесь нужен второй.
using Tensor = AI.ML.NeuralNetworks.V2.Tensor;

namespace FAI.Router.Training;

/// <summary>
/// Перевод между алгеброй роутера (Vector, Matrix) и тензорами автограда.
/// Тензоры считают только во float32, поэтому точность на границе теряется — для обучения
/// весов это несущественно, но обратная запись всегда идёт через этот же мост, чтобы
/// расхождение не накапливалось незаметно.
/// </summary>
internal static class TensorBridge
{
    /// <summary>
    /// Вектор как строка (1, n) — в таком виде его берут MatMul и CosineSimilarity
    /// </summary>
    public static Tensor ToRow(Vector vector) => Tensor.From(ToFloats(vector), new Shape(1, vector.Count));

    /// <summary>
    /// Вектор как колонка (n, 1) — правый множитель матрицы
    /// </summary>
    public static Tensor ToColumn(Vector vector) => Tensor.From(ToFloats(vector), new Shape(vector.Count, 1));

    /// <summary>
    /// Матрица как тензор (строк, столбцов)
    /// </summary>
    public static Tensor ToTensor(Matrix matrix)
    {
        float[] values = new float[matrix.Height * matrix.Width];

        for (int row = 0; row < matrix.Height; row++)
            for (int column = 0; column < matrix.Width; column++)
                values[row * matrix.Width + column] = (float)matrix[row, column];

        return Tensor.From(values, new Shape(matrix.Height, matrix.Width));
    }

    /// <summary>
    /// Переносит обученные значения обратно в вектор
    /// </summary>
    public static void WriteBack(Tensor tensor, Vector target)
    {
        ReadOnlySpan<float> values = tensor.AsReadOnlySpan<float>();

        for (int i = 0; i < target.Count; i++)
            target[i] = values[i];
    }

    /// <summary>
    /// Переносит обученные значения обратно в матрицу
    /// </summary>
    public static void WriteBack(Tensor tensor, Matrix target)
    {
        ReadOnlySpan<float> values = tensor.AsReadOnlySpan<float>();

        for (int row = 0; row < target.Height; row++)
            for (int column = 0; column < target.Width; column++)
                target[row, column] = values[row * target.Width + column];
    }

    /// <summary>
    /// Единственное число из скалярного тензора (значение функции потерь)
    /// </summary>
    public static double Scalar(Tensor tensor) => tensor.AsReadOnlySpan<float>()[0];

    private static float[] ToFloats(Vector vector)
    {
        float[] values = new float[vector.Count];

        for (int i = 0; i < vector.Count; i++)
            values[i] = (float)vector[i];

        return values;
    }
}
