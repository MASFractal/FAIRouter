using System.Text.Json;
using System.Text.RegularExpressions;

namespace FAI.Router.Catalog;

/// <summary>
/// Разбор страниц, которые сервер отдает разметкой RSC: данные лежат JSON внутри кусков
/// <c>self.__next_f.push([1,"..."])</c>. Так устроены и арена, и Artificial Analysis.
/// </summary>
internal static partial class Rsc
{
    internal const string UserAgent = "Mozilla/5.0 (FAIRouter)";

    /// <summary>Текст всех кусков страницы подряд</summary>
    public static string Payload(string html) =>
        string.Concat(Chunk().Matches(html).Select(match => Decode(match.Groups[1].Value)));

    /// <summary>Массив JSON от позиции открывающей скобки до парной ей, с учетом строк</summary>
    public static string ArrayAt(string text, int start)
    {
        int depth = 0;
        bool inString = false;

        for (int i = start; i < text.Length; i++)
        {
            char c = text[i];

            if (inString)
            {
                if (c == '\\') i++;
                else if (c == '"') inString = false;
            }
            else if (c == '"') inString = true;
            else if (c == '[') depth++;
            else if (c == ']' && --depth == 0) return text[start..(i + 1)];
        }

        throw new FormatException("Массив JSON на странице не закрыт.");
    }

    /// <summary>Страница по адресу с обычным заголовком браузера</summary>
    public static async Task<string> GetAsync(string url, HttpClient? client, CancellationToken cancellationToken)
    {
        using HttpClient? own = client is null ? new HttpClient() : null;
        using HttpRequestMessage request = new(HttpMethod.Get, url);
        request.Headers.TryAddWithoutValidation("User-Agent", UserAgent);

        using HttpResponseMessage response = await (client ?? own!).SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        return await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
    }

    // Кусок это строковый литерал JS; правила экранирования те же, что у JSON
    private static string Decode(string chunk)
    {
        try
        {
            return JsonSerializer.Deserialize<string>($"\"{chunk}\"") ?? "";
        }
        catch (JsonException)
        {
            return Regex.Unescape(chunk);
        }
    }

    [GeneratedRegex("""self\.__next_f\.push\(\[1,"(.*?)"\]\)""", RegexOptions.Singleline)]
    private static partial Regex Chunk();
}
