using System.Net;
using System.Net.Sockets;
using System.Text;
using AI.LLM.Core.Models.Common.Messages;
using FAI.Router.LLM;

namespace FAI.Router.Tests;

/// <summary>Обращение к поставщику по сети: окончательный отказ не повторяется</summary>
public class ProviderHttpTests
{
    /// <summary>
    /// Отказ 400 уходит в сеть один раз, хотя движок повторяет любой сбой, и доходит до
    /// вызывающего своим типом с кодом, а не обычным исключением с текстом запроса
    /// </summary>
    [Fact]
    public async Task Final_rejection_is_not_repeated()
    {
        using TcpListener listener = new(IPAddress.Loopback, 0);
        listener.Start();
        int port = ((IPEndPoint)listener.LocalEndpoint).Port;
        int requests = 0;

        Task server = Task.Run(async () =>
        {
            while (true)
            {
                using TcpClient client = await listener.AcceptTcpClientAsync();
                Interlocked.Increment(ref requests);
                await RejectAsync(client.GetStream());
            }
        });

        StyleClassifier classifier = new(new OpenAiCompatibleLlm($"http://127.0.0.1:{port}/v1", "rtr_test", "m"));

        ProviderRejectedException error = await Assert.ThrowsAsync<ProviderRejectedException>(() => classifier.AssessAsync("текст"));

        Assert.Equal(400, error.Code);
        Assert.DoesNotContain("текст", error.Message);
        Assert.Equal(1, Volatile.Read(ref requests));
        listener.Stop();
    }

    // Читает заголовки и тело запроса и отвечает отказом 400
    private static async Task RejectAsync(NetworkStream stream)
    {
        byte[] buffer = new byte[64 * 1024];
        StringBuilder head = new();
        int read;

        while (!head.ToString().Contains("\r\n\r\n") && (read = await stream.ReadAsync(buffer)) > 0)
            head.Append(Encoding.UTF8.GetString(buffer, 0, read));

        string text = head.ToString();
        int length = int.Parse(System.Text.RegularExpressions.Regex.Match(text, @"Content-Length:\s*(\d+)", System.Text.RegularExpressions.RegexOptions.IgnoreCase).Groups[1].Value);
        int received = Encoding.UTF8.GetByteCount(text[(text.IndexOf("\r\n\r\n", StringComparison.Ordinal) + 4)..]);

        while (received < length && (read = await stream.ReadAsync(buffer)) > 0)
            received += read;

        byte[] body = Encoding.UTF8.GetBytes("""{"error":{"message":"bad request"}}""");
        byte[] response = Encoding.UTF8.GetBytes($"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\nContent-Length: {body.Length}\r\nConnection: close\r\n\r\n");
        await stream.WriteAsync(response);
        await stream.WriteAsync(body);
    }
}
