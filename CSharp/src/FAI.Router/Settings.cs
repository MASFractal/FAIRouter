using System;
using System.Collections.Generic;
using System.Text;

namespace FAI.Router;

public class Settings
{
    public static double WQ { get; set; } = 0.5;
    public static double Wt { get; set; } = 0.25;
    public static double WC { get; set; } = 0.25;

    /// <summary>
    /// Размерность пространства признаков
    /// </summary>
    public static int FeaturesDim { get; set; } = 3;
}
