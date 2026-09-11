using AI.ML.NeuralNetworks.V2;
using AI.ML.NeuralNetworks.V2.Losses;
using AI.ML.NeuralNetworks.V2.Nn;
using AI.ML.NeuralNetworks.V2.Optim;
using FAI.Router.JudgeLogic;

namespace FAI.Router.Training;

/// <summary>
/// Обучение судьи: матрица трансформации подгоняется так, чтобы оценка судьи повторяла
/// оценку человека. Ошибкой служит квадрат расхождения, а обучение идет спуском по градиенту.
/// </summary>
public class JudgeTrainer
{
    private readonly Judge _judge;
    private readonly float _learningRate;

    /// <summary>
    /// Обучение судьи по оценкам человека
    /// </summary>
    /// <param name="judge">Судья, чья матрица обучается</param>
    /// <param name="learningRate">Скорость обучения</param>
    public JudgeTrainer(Judge judge, float learningRate = 0.01f)
    {
        _judge = judge;
        _learningRate = learningRate;
    }

    /// <summary>
    /// Один шаг обучения. Возвращает ошибку до шага, по ее убыванию видно, что судья учится.
    /// </summary>
    /// <param name="requested">Заказанная спецификация (ТЗ)</param>
    /// <param name="actual">Фактическая спецификация ответа</param>
    /// <param name="humanScore">Оценка человека: единица означает «нравится», ноль означает «нет»</param>
    public double Train(Specifications requested, Specifications actual, double humanScore)
    {
        int dim = Settings.FeaturesSpecDim;

        // Тензор строится на шаг от НЫНЕШНЕЙ матрицы судьи, а не копируется один раз в
        // конструкторе. Копия переставала быть матрицей судьи после первой же загрузки весов:
        // Load подменяет объект матрицы, и первый шаг обучения перезаписывал загруженное
        // единичной матрицей с одним шагом. Спуск без импульса состояния между шагами не
        // держит, поэтому терять здесь нечего.
        Parameter transformer = new(TensorBridge.ToTensor(_judge.TransformerW));
        Optimizer optimizer = new SGD([transformer], lr: _learningRate);

        Tensor request = TensorBridge.ToColumn(requested.FeaturesSpecificationVector);
        Tensor fact = TensorBridge.ToRow(actual.FeaturesSpecificationVector);
        Tensor target = Tensor.From([(float)humanScore], new Shape(1));

        optimizer.ZeroGrad();

        Tensor transformed = transformer.Tensor.MatMul(request).Reshape(1, dim);
        Tensor score = EmbeddingLosses.CosineSimilarity(fact, transformed);
        Tensor loss = RegressionLosses.MSE(score, target);

        loss.Backward();
        optimizer.Step();

        // Судья считает по Matrix, а не по тензору: без обратной записи обучение
        // никак не отразилось бы на его оценках
        TensorBridge.WriteBack(transformer.Tensor, _judge.TransformerW);

        return TensorBridge.Scalar(loss);
    }
}
