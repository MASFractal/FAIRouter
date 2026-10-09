using FAI.Router.JudgeLogic;
using FAI.Router.RoutedElements;
using static FAI.Router.Tests.Scene;

namespace FAI.Router.Tests;

/// <summary>Исполнение хода: запасной кандидат, отмена, пустой ответ, распознавание своим распознавателем</summary>
public class ExecutionTests
{
    /// <summary>Таймаут клиента приходит исключением отмены, но отменой вызывающего не является: работу берет запасной</summary>
    [Fact]
    public async Task Timeout_falls_back_to_the_next_candidate()
    {
        var features = Features();
        var leader = Element(features, "лидер", 0.9, 1);
        var spare = Element(features, "запасной", 0.5, 1);
        var trace = Trace(features, leader, spare);

        string text = await Env.ExecuteAsync(trace, candidate => candidate == leader
            ? Task.FromException<string>(new TaskCanceledException("таймаут клиента"))
            : Task.FromResult("ответ"));

        Assert.Equal("ответ", text);
        Assert.Same(spare, trace.Winner);
        Assert.Equal(["лидер"], trace.Failed);
        Assert.True(trace.IsExploration);
        Assert.Equal(spare.GetQualityScore(trace.InputFeatureVector), trace.Forecast);
    }

    [Fact]
    public async Task Caller_cancellation_stops_the_turn()
    {
        var features = Features();
        var leader = Element(features, "лидер", 0.9, 1);
        var spare = Element(features, "запасной", 0.5, 1);
        using CancellationTokenSource cancel = new();
        int calls = 0;

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Env.ExecuteAsync(Trace(features, leader, spare), _ =>
        {
            calls++;
            cancel.Cancel();
            return Task.FromException<string>(new OperationCanceledException(cancel.Token));
        }, cancel.Token));

        Assert.Equal(1, calls);
    }

    [Fact]
    public async Task Empty_answer_is_a_failure()
    {
        var features = Features();
        var leader = Element(features, "лидер", 0.9, 1);
        var spare = Element(features, "запасной", 0.5, 1);
        var trace = Trace(features, leader, spare);

        string text = await Env.ExecuteAsync(trace, candidate => Task.FromResult(candidate == leader ? "  " : "ответ"));

        Assert.Equal("ответ", text);
        Assert.Equal(["лидер"], trace.Failed);
    }

    /// <summary>Свой распознаватель вместо общего Settings.LLM, объем входа всего диалога и отмена до обращения к модели</summary>
    [Fact]
    public async Task Route_uses_its_own_recognizer_and_dialog_volume()
    {
        var features = Features();
        FixedSpecs specs = new(new Specifications { SymbolLength = 3000 });

        var trace = await Env.RouteAsync("Напиши обзор", [Element(features, "a", 0.5, 1), Element(features, "b", 0.6, 1)],
            specs: specs, inputTokens: 5000, random: new Random(1));

        Assert.Equal(1, specs.Calls);
        Assert.NotNull(trace.Forecast);

        using CancellationTokenSource cancel = new();
        cancel.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            Env.RouteAsync("Напиши обзор", [Element(features, "a", 0.5, 1)], specs: specs, cancellationToken: cancel.Token));
        Assert.Equal(1, specs.Calls);
    }

    /// <summary>Объем входа по всему диалогу двигает цену: длинный диалог дорожает у всех кандидатов</summary>
    [Fact]
    public void Dialog_volume_enters_the_cost()
    {
        var features = Features();
        BaseRoutedElement element = Element(features, "a", 0.5, 1);
        double shortCost = element.GetCost(features);

        features.InputLen = 50_000;

        Assert.True(element.GetCost(features) > shortCost);
    }
}
