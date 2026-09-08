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
    private readonly Parameter _transformer;
    private readonly Optimizer _optimizer;

    /// <summary>
    /// Обучение судьи по оценкам человека
    /// </summary>
    /// <param name="judge">Судья, чья матрица обучается</param>
    /// <param name="learningRate">Скорость обучения</param>
    public JudgeTrainer(Judge judge, float learningRate = 0.01f)
    {
        _judge = judge;
        _transformer = new Parameter(TensorBridge.ToTensor(judge.TransformerW));
        _optimizer = new SGD([_transformer], lr: learningRate);
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

        Tensor request = TensorBridge.ToColumn(requested.FeaturesSpecificationVector);
        Tensor fact = TensorBridge.ToRow(actual.FeaturesSpecificationVector);
        Tensor target = Tensor.From([(float)humanScore], new Shape(1));

        _optimizer.ZeroGrad();

        Tensor transformed = _transformer.Tensor.MatMul(request).Reshape(1, dim);
        Tensor score = EmbeddingLosses.CosineSimilarity(fact, transformed);
        Tensor loss = RegressionLosses.MSE(score, target);

        loss.Backward();
        _optimizer.Step();

        // Судья считает по Matrix, а не по тензору: без обратной записи обучение
        // никак не отразилось бы на его оценках
        TensorBridge.WriteBack(_transformer.Tensor, _judge.TransformerW);

        return TensorBridge.Scalar(loss);
    }
}
