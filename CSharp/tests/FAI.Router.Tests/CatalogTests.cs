using System.Text.Json;
using FAI.Router.Catalog;
using FAI.Router.Enums;
using FAI.Router.Training;

namespace FAI.Router.Tests;

/// <summary>Каталог поставщика, кандидаты из него, снимок замеров и сопоставление имен</summary>
public class CatalogTests
{
    /// <summary>Битая запись пропускается, а не роняет весь каталог; цена без данных неизвестна, а не бесплатна</summary>
    [Fact]
    public void OpenRouter_catalog_survives_broken_items()
    {
        const string json =
            """
            {"data":[
              {"id":"ok/model","name":"Ok","pricing":{"prompt":"0.000001","completion":0.000002,"web_search":"0.01"},
               "context_length":128000.5,"top_provider":{"max_completion_tokens":"4096"},
               "architecture":{"input_modalities":null},"supported_parameters":["tools",5]},
              {"name":"без идентификатора"},
              {"id":"free/model","pricing":{"prompt":"0","completion":"0"}},
              {"id":"auto/model","pricing":{"prompt":"-1","completion":"-1"}},
              {"id":"noprice/model","pricing":null,"architecture":"строка"},
              "мусор", null
            ]}
            """;

        Dictionary<string, ModelInfo> models = ModelCatalog.Parse(json).ToDictionary(model => model.Id);

        Assert.Equal(["ok/model", "free/model", "auto/model", "noprice/model"], models.Keys);

        ModelInfo ok = models["ok/model"];
        Assert.Equal(1, ok.DollarsPerMillionInput, 9);
        Assert.Equal(2, ok.DollarsPerMillionOutput, 9);
        Assert.Equal(128000, ok.ContextTokens);
        Assert.Equal(4096, ok.MaxAnswerTokens);
        Assert.Equal(Capability.Code | Capability.Formulas | Capability.Tools | Capability.WebSearch, ok.Capabilities);

        Assert.Equal(0, models["free/model"].DollarsPerMillionInput);
        Assert.True(models["auto/model"].DollarsPerMillionInput < 0);
        Assert.True(models["noprice/model"].DollarsPerMillionOutput < 0);
        Assert.DoesNotContain("noprice/model", ModelCatalog.Select([ModelCatalog.All], models).Ids);
    }

    [Fact]
    public void FractalRouter_catalog_survives_broken_items()
    {
        const string json =
            """
            [
              {"id":"a/model","pricing":{"prompt_rub_per_1m":"12,5","completion_rub_per_1m":50},"modalities":["text","image"],"max_completion_tokens":2048.7},
              {"id":"b/model","pricing":"нет","modalities":null},
              {"id":17}
            ]
            """;

        Dictionary<string, ModelInfo> models = ModelCatalog.ParseFractalRouter(json).ToDictionary(model => model.Id);

        Assert.Equal(12.5, models["a/model"].DollarsPerMillionInput, 9);
        Assert.Equal(2048, models["a/model"].MaxAnswerTokens);
        Assert.True(models["a/model"].Capabilities.HasFlag(Capability.Vision));
        Assert.True(models["b/model"].DollarsPerMillionInput < 0);
        Assert.Equal(3, models.Count);
    }

    /// <summary>
    /// Цена из prices поверх каталога не стирает возможностей модели; модель только из prices считается
    /// способной на все; повтор в списке не роняет роутер
    /// </summary>
    [Fact]
    public void Candidates_keep_catalog_capabilities_and_skip_repeats()
    {
        Dictionary<string, ModelInfo> known = new()
        {
            ["a/model"] = new("a/model", "A", 1, 2, 100_000, 4_000, Capability.Code | Capability.Vision),
            ["auto/model"] = new("auto/model", "Auto", -1, -1, 0, 0, Capability.Code),
        };
        Dictionary<string, Price> prices = new() { ["a/model"] = new Price(3, 4), ["x/model"] = new Price(5, 6) };

        var candidates = ModelCatalog.CreateCandidates(["a/model", "a/model", "x/model", "auto/model"], known, prices, null, null, null);

        Assert.Equal(["a/model", "x/model", "auto/model"], candidates.Select(candidate => candidate.Name));
        Assert.Equal(Capability.Code | Capability.Vision, candidates[0].Capabilities);
        Assert.Equal(3, candidates[0].DPMTInp);
        Assert.Equal(100_000, candidates[0].ContextWindow);
        Assert.Equal(Capability.All, candidates[1].Capabilities);
        Assert.False(candidates[2].HasKnownPrice);
    }

    /// <summary>
    /// Модель без рейтингов получает уровень поля, а не случайный вектор; скорость вне серии это нижняя
    /// граница серии, а не 50
    /// </summary>
    [Fact]
    public void Unrated_model_starts_from_the_field_level()
    {
        var snapshot = BenchmarkSnapshot.LoadEmbedded();
        ModelInfo unknown = new("vendor/never-rated-model-x", "X", 1, 2, 0, 0, Capability.All);

        var first = ModelCatalog.CreateElement(unknown, null, snapshot, null);
        var second = ModelCatalog.CreateElement(unknown, null, snapshot, null);

        Assert.Equal(Scene.Values(first.IdealMatchVector), Scene.Values(second.IdealMatchVector));
        Assert.Equal(0, first.PriorExperience);
        Assert.Equal(snapshot.Entries["bench:speed"].Min(row => row.Score), first.TPS, 9);
        Assert.Equal(Scene.Values(BenchmarkPrior.Uniform(BenchmarkPrior.FieldQuality(snapshot)!.Value, null)),
            Scene.Values(first.IdealMatchVector));
    }

    [Fact]
    public async Task Network_failure_falls_back_to_bundled_prices()
    {
        var models = await ModelCatalog.FetchOrPopularAsync(_ => throw new HttpRequestException("нет сети"));

        Assert.NotEmpty(models);
        Assert.Contains(models, model => model.DollarsPerMillionInput > 0 && model.DollarsPerMillionOutput > 0);
        Assert.Equal(ModelCatalog.PopularModels(), models.Select(model => model.Id));
    }

    /// <summary>
    /// Хвосты max и thinking у одних поставщиков означают другой продукт: точное имя их хранит, семейство
    /// срезает. У Claude размышление это режим той же модели
    /// </summary>
    [Theory]
    [InlineData("qwen/qwen3.8-max", "qwen/qwen3.8")]
    [InlineData("openai/gpt-5.1-codex-max", "openai/gpt-5.1-codex")]
    [InlineData("moonshotai/kimi-k2-thinking", "moonshotai/kimi-k2")]
    public void Product_suffixes_stay_apart(string product, string base_)
    {
        Assert.NotEqual(ModelNames.Canonical(product), ModelNames.Canonical(base_));
        Assert.Equal(ModelNames.Canonical(product, relaxed: true), ModelNames.Canonical(base_, relaxed: true));
    }

    [Fact]
    public void Claude_thinking_is_the_same_model()
    {
        Assert.Equal(ModelNames.Canonical("anthropic/claude-opus-4.7"), ModelNames.Canonical("claude-opus-4-7-thinking-32k"));
        Assert.Equal(ModelNames.Canonical("anthropic/claude-opus-4.1"), ModelNames.Canonical("claude-opus-4-1-20250805-thinking-16k"));
    }

    /// <summary>Сначала точное совпадение, потом семейство: продукт не получает чужих оценок, пока есть свои</summary>
    [Fact]
    public void Find_prefers_exact_names_then_family()
    {
        BenchmarkEntry thinking = new("kimi-k2-thinking", "Kimi K2 Thinking", "Moonshot", 50);
        BenchmarkEntry plain = new("kimi-k2", "Kimi K2", "Moonshot", 40);

        Assert.Same(thinking, ModelNames.Find("moonshotai/kimi-k2-thinking", [plain, thinking]));
        Assert.Same(plain, ModelNames.Find("moonshotai/kimi-k2", [plain, thinking]));
        Assert.Same(thinking, ModelNames.Find("moonshotai/kimi-k2", [thinking]));

        BenchmarkSnapshot snapshot = new() { Entries = { ["pref:text/overall"] = [plain, thinking] } };
        Assert.Same(thinking, snapshot.Find("pref:text/overall", "moonshotai/kimi-k2-thinking"));
        Assert.Same(plain, snapshot.Find("pref:text/overall", "moonshotai/kimi-k2"));
    }

    /// <summary>Индекс серии и перебор отвечают одинаково на встроенном снимке</summary>
    [Fact]
    public void Indexed_find_matches_the_scan()
    {
        var snapshot = BenchmarkSnapshot.LoadEmbedded();
        string[] names = [.. snapshot.Entries.Values.SelectMany(rows => rows).Select(row => "vendor/" + row.ModelKey).Distinct().Take(300)];

        foreach ((string key, List<BenchmarkEntry> rows) in snapshot.Entries.Take(20))
            foreach (string name in names)
                Assert.Same(ModelNames.Find(name, rows), snapshot.Find(key, name));
    }

    /// <summary>Обрезка снимка хранит границы полной серии: доля качества по обрезку та же, что по полной серии</summary>
    [Fact]
    public void Top_keeps_full_series_bounds()
    {
        BenchmarkSnapshot full = new()
        {
            FetchedAt = "2026-10-01T00:00:00Z",
            Entries = { ["pref:text/overall"] = [new("a", "a", "", 1500), new("b", "b", "", 1490), new("c", "c", "", 1200), new("d", "d", "", 1100)] },
        };

        BenchmarkSnapshot cut = full.Top(2);

        Assert.Equal(2, cut.Entries["pref:text/overall"].Count);
        Assert.Equal(full.Quality("pref:text/overall", "vendor/b"), cut.Quality("pref:text/overall", "vendor/b"));
        Assert.Equal(0.975, cut.Quality("pref:text/overall", "vendor/b")!.Value, 9);
        Assert.Equal(4, cut.Bounds!["pref:text/overall"].Count);

        // Уровень поля по средней полной серии, а не по верхним строкам
        Assert.Equal((1322.5 - 1100) / 400, cut.MeanShare("pref:text/overall")!.Value.Share, 9);

        // Формат обратно совместим: снимок без границ пишется без поля, с границами читается обратно
        Assert.DoesNotContain("bounds", JsonSerializer.Serialize(full));
        BenchmarkSnapshot loaded = JsonSerializer.Deserialize<BenchmarkSnapshot>(JsonSerializer.Serialize(cut))!;
        Assert.Equal(cut.Quality("pref:text/overall", "vendor/b"), loaded.Quality("pref:text/overall", "vendor/b"));
    }
}
