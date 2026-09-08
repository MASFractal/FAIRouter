using Microsoft.Data.Sqlite;

namespace FAI.Router.Persistence;

/// <summary>
/// Открытие базы: веса и накопленный опыт лежат в одном файле, поэтому и путь к нему
/// превращается в соединение одинаково
/// </summary>
internal static class SqliteDb
{
    /// <summary>
    /// Открытое соединение с базой; файл создается при отсутствии
    /// </summary>
    /// <param name="databasePath">Путь к файлу базы</param>
    public static SqliteConnection Open(string databasePath)
    {
        SqliteConnection connection = new(new SqliteConnectionStringBuilder { DataSource = databasePath }.ToString());
        connection.Open();

        return connection;
    }
}
