using System.Net;
using System.Net.Http.Headers;
using System.Text.Json;
using AI.LLM.API.LLMAPI;
using AI.LLM.Clients.OpenRouter;
using AI.LLM.Infrastructure.Http;
using AI.LLM.Services.LLM;

namespace FAI.Router.Tests;

/// <summary>
/// Модель без сети: отвечает заготовленными ответами потоком SSE, как поставщик, и запоминает
/// запросы. Движок (ChatLLMApi) идет своим обычным путем, поэтому проверяется и разбор ответа.
/// </summary>
internal sealed class ScriptedModel : IWebAPIClient
{
    private readonly Queue<(string Content, string Finish)> _replies;
    private readonly TimeSpan _delay;

    public ScriptedModel(params (string Content, string Finish)[] replies) : this(TimeSpan.Zero, replies)
    {
    }

    public ScriptedModel(TimeSpan delay, params (string Content, string Finish)[] replies)
    {
        _delay = delay;
        _replies = new(replies);
    }

    /// <summary>Тела запросов по порядку</summary>
    public List<string> Requests { get; } = [];

    public AuthenticationHeaderValue? Authentication { get; set; }

    /// <summary>Клиент модели поверх этой заготовки</summary>
    public LLMBase Client => new(new OpenRouterModelApi(this, "scripted/judge"));

    public async Task<HttpResponseMessage> PostAsJsonAsync(string apiUrl, SendDataLLM sendData, CancellationToken? concelationToken = default)
    {
        CancellationToken token = concelationToken ?? CancellationToken.None;
        Requests.Add(sendData.GetJson());

        if (_delay > TimeSpan.Zero)
            await Task.Delay(_delay, token);

        token.ThrowIfCancellationRequested();
        (string content, string finish) = _replies.Dequeue();
        string chunk = JsonSerializer.Serialize(new { choices = new[] { new { index = 0, delta = new { content }, finish_reason = finish } } });

        return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent($"data: {chunk}\n\ndata: [DONE]\n\n") };
    }

    public void Dispose()
    {
    }
}
