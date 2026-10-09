using System.Net;
using System.Net.Http.Headers;
using System.Runtime.CompilerServices;
using System.Text;
using AI.LLM.API.LLMAPI;
using AI.LLM.Infrastructure.Http;

namespace FAI.Router.LLM;

/// <summary>
/// Отправка запросов поставщику через одно соединение на процесс. Окончательный отказ поставщика
/// (4xx, кроме таймаута и лимита частоты) не повторяется, а пауза из Retry-After соблюдается.
/// </summary>
/// <remarks>
/// Прежде каждый клиент модели создавал свой HttpClient со своим обработчиком, и вытесненных
/// клиентов никто не освобождал. Повторы ведет движок (ChatLLMApi): две попытки на любой сбой,
/// включая 400 и 401. Повтор он отправляет тем же объектом запроса, поэтому отказ запоминается на
/// этом объекте и второй раз в сеть не уходит, а пауза из Retry-After выжидается перед повтором.
/// </remarks>
internal sealed class ProviderHttp : IWebAPIClient
{
    /// <summary>Дольше этого Retry-After не выжидается: ход ждет человек</summary>
    private static readonly TimeSpan MaxRetryAfter = TimeSpan.FromSeconds(30);

    /// <summary>Сколько ждать заголовков ответа: молчащий сервер не держит ход до общего таймаута</summary>
    private static readonly TimeSpan HeadersTimeout = TimeSpan.FromSeconds(70);

    // Срок жизни соединения ограничен, чтобы смена адресов поставщика в DNS доходила до процесса
    private static readonly HttpClient Shared = new(new SocketsHttpHandler
    {
        ConnectTimeout = TimeSpan.FromSeconds(30),
        PooledConnectionLifetime = TimeSpan.FromMinutes(10),
        PooledConnectionIdleTimeout = TimeSpan.FromSeconds(60),
        MaxConnectionsPerServer = 32,
    })
    {
        Timeout = Timeout.InfiniteTimeSpan
    };

    private readonly string _apiKey;
    private readonly ConditionalWeakTable<SendDataLLM, ProviderRejectedException> _rejected = new();
    private readonly ConditionalWeakTable<SendDataLLM, StrongBox<DateTimeOffset>> _notBefore = new();

    /// <param name="apiKey">Ключ поставщика, уходит заголовком Bearer</param>
    public ProviderHttp(string apiKey) => _apiKey = apiKey;

    /// <inheritdoc />
    public AuthenticationHeaderValue? Authentication { get; set; }

    /// <inheritdoc />
    public async Task<HttpResponseMessage> PostAsJsonAsync(string apiUrl, SendDataLLM sendData, CancellationToken? concelationToken = default)
    {
        CancellationToken cancellationToken = concelationToken ?? CancellationToken.None;

        if (_rejected.TryGetValue(sendData, out ProviderRejectedException? rejected))
            throw rejected;

        if (_notBefore.TryGetValue(sendData, out StrongBox<DateTimeOffset>? notBefore) && notBefore.Value > DateTimeOffset.UtcNow)
            await Task.Delay(notBefore.Value - DateTimeOffset.UtcNow, cancellationToken).ConfigureAwait(false);

        using HttpRequestMessage request = new(HttpMethod.Post, apiUrl)
        {
            Content = new StringContent(sendData.GetJson(), Encoding.UTF8, "application/json")
        };

        if (!string.IsNullOrEmpty(_apiKey))
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", _apiKey);

        using CancellationTokenSource headers = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        headers.CancelAfter(HeadersTimeout);
        HttpResponseMessage response;

        try
        {
            response = await Shared.SendAsync(request,
                sendData.Stream is true ? HttpCompletionOption.ResponseHeadersRead : HttpCompletionOption.ResponseContentRead,
                headers.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            throw new TimeoutException($"Поставщик не ответил за {HeadersTimeout.TotalSeconds:0} с.");
        }

        if (IsFinal(response.StatusCode))
        {
            string body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            response.Dispose();
            ProviderRejectedException error = new((int)response.StatusCode, body);
            _rejected.AddOrUpdate(sendData, error);
            throw error;
        }

        if (RetryAfter(response) is { } delay)
            _notBefore.AddOrUpdate(sendData, new StrongBox<DateTimeOffset>(DateTimeOffset.UtcNow + delay));

        return response;
    }

    /// <summary>Соединение общее на процесс и живет вместе с ним</summary>
    public void Dispose()
    {
    }

    // Окончательный отказ: запрос неверен, ключ не годится, денег нет, модели нет. Таймаут, конфликт,
    // ранний запрос и лимит частоты проходят повтором
    private static bool IsFinal(HttpStatusCode status) =>
        (int)status is >= 400 and < 500
        && status is not (HttpStatusCode.RequestTimeout or HttpStatusCode.Conflict or (HttpStatusCode)425 or HttpStatusCode.TooManyRequests);

    private static TimeSpan? RetryAfter(HttpResponseMessage response)
    {
        RetryConditionHeaderValue? header = response.Headers.RetryAfter;
        TimeSpan? delay = header?.Delta ?? (header?.Date is { } date ? date - DateTimeOffset.UtcNow : null);

        return delay is { } value && value > TimeSpan.Zero ? (value < MaxRetryAfter ? value : MaxRetryAfter) : null;
    }
}

/// <summary>
/// Поставщик окончательно отказал (4xx): повтор того же запроса ничего не изменит
/// </summary>
public sealed class ProviderRejectedException : HttpRequestException
{
    private const int MessageChars = 300;

    /// <summary>Поставщик окончательно отказал</summary>
    /// <param name="statusCode">Код ответа</param>
    /// <param name="providerMessage">Текст ответа поставщика</param>
    public ProviderRejectedException(int statusCode, string? providerMessage)
        : base($"Поставщик отказал, код {statusCode}.", null, (HttpStatusCode)statusCode)
    {
        ProviderMessage = providerMessage is { Length: > MessageChars } text ? text[..MessageChars] : providerMessage ?? "";
    }

    /// <summary>Код ответа поставщика числом</summary>
    public int Code => (int)(StatusCode ?? 0);

    /// <summary>Начало ответа поставщика: причина отказа. Текст запроса поставщики в нем не повторяют.</summary>
    public string ProviderMessage { get; }

    /// <summary>Отказ поставщика в цепочке вложенных исключений движка; пусто, если его там нет</summary>
    /// <param name="error">Исключение движка</param>
    public static ProviderRejectedException? Find(Exception? error)
    {
        for (; error is not null; error = error.InnerException)
            if (error is ProviderRejectedException rejected)
                return rejected;

        return null;
    }
}
