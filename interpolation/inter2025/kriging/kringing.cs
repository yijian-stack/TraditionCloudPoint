using System;
using System.Collections.Generic;
using System.Linq;

namespace kriging.PointCloudInterpolation
{
    /// <summary>
    /// 普通克里金插值类，用于对点云数据进行空间插值。
    /// </summary>
    public class OrdinaryKriging
    {
        private const double Epsilon = 1.0e-10;
        private readonly double[] xOriginal;
        private readonly double[] yOriginal;
        private readonly double[] zValues;
        private readonly string variogramModel;
        private readonly double[] variogramModelParameters;
        private readonly double anisotropyScaling;
        private readonly double anisotropyAngle;
        private readonly double xCenter;
        private readonly double yCenter;
        private readonly double[] xAdjusted;
        private readonly double[] yAdjusted;
        private readonly double c0; // 协方差在零距离处的值

        /// <summary>
        /// 初始化普通克里金插值对象。
        /// </summary>
        /// <param name="x">X坐标数组。</param>
        /// <param name="y">Y坐标数组。</param>
        /// <param name="z">Z值数组。</param>
        /// <param name="variogramModel">变异函数模型（默认为 "spherical"）。</param>
        /// <param name="variogramParameters">变异函数参数（可选）。</param>
        /// <param name="anisotropyScaling">各向异性缩放因子（默认为 1.0）。</param>
        /// <param name="anisotropyAngle">各向异性角度（默认为 0.0）。</param>
        public OrdinaryKriging(double[] x, double[] y, double[] z, string variogramModel = "spherical", double[] variogramParameters = null, double anisotropyScaling = 1.0, double anisotropyAngle = 0.0)
        {
            if (x.Length != y.Length || x.Length != z.Length)
                throw new ArgumentException("输入数组 x, y, z 必须具有相同的长度。");

            xOriginal = x;
            yOriginal = y;
            zValues = z;
            this.variogramModel = variogramModel;

            xCenter = (x.Max() + x.Min()) / 2.0;
            yCenter = (y.Max() + y.Min()) / 2.0;

            this.anisotropyScaling = anisotropyScaling;
            this.anisotropyAngle = anisotropyAngle;
            var adjustedPoints = AdjustForAnisotropy(x, y, xCenter, yCenter, anisotropyScaling, anisotropyAngle);
            xAdjusted = adjustedPoints.Item1;
            yAdjusted = adjustedPoints.Item2;

            variogramModelParameters = InitializeVariogramModel(variogramModel, variogramParameters, xAdjusted, yAdjusted, z);
            c0 = CalculateC0(variogramModel, variogramModelParameters);
        }

        /// <summary>
        /// 计算零距离处的协方差值。
        /// </summary>
        /// <param name="model">变异函数模型。</param>
        /// <param name="parameters">变异函数参数。</param>
        /// <returns>零距离处的协方差值。</returns>
        private double CalculateC0(string model, double[] parameters)
        {
            switch (model)
            {
                case "spherical":
                case "exponential":
                case "gaussian":
                    return parameters[0] + parameters[2]; // 基台值 + 残差
                case "linear":
                    return parameters[0]; // 斜率 * 距离 + 残差（无基台值）
                case "power":
                    return parameters[2]; // 残差
                default:
                    return 0;
            }
        }

        /// <summary>
        /// 调整各向异性的点坐标。
        /// </summary>
        /// <param name="x">X坐标数组。</param>
        /// <param name="y">Y坐标数组。</param>
        /// <param name="xCenter">X中心坐标。</param>
        /// <param name="yCenter">Y中心坐标。</param>
        /// <param name="scaling">各向异性缩放因子。</param>
        /// <param name="angle">各向异性角度（弧度）。</param>
        /// <returns>调整后的X和Y坐标。</returns>
        private Tuple<double[], double[]> AdjustForAnisotropy(double[] x, double[] y, double xCenter, double yCenter, double scaling, double angle)
        {
            double[] xAdjusted = new double[x.Length];
            double[] yAdjusted = new double[y.Length];

            double angleRad = angle * Math.PI / 180.0;
            double cosTheta = Math.Cos(angleRad);
            double sinTheta = Math.Sin(angleRad);

            for (int i = 0; i < x.Length; i++)
            {
                double dx = x[i] - xCenter;
                double dy = y[i] - yCenter;

                double rotatedX = dx * cosTheta - dy * sinTheta;
                double rotatedY = dx * sinTheta + dy * cosTheta;

                xAdjusted[i] = rotatedX / scaling + xCenter;
                yAdjusted[i] = rotatedY * scaling + yCenter;
            }

            return Tuple.Create(xAdjusted, yAdjusted);
        }

        /// <summary>
        /// 初始化变异函数模型。
        /// </summary>
        /// <param name="model">变异函数模型。</param>
        /// <param name="parameters">变异函数参数（可选）。</param>
        /// <param name="x">X坐标数组。</param>
        /// <param name="y">Y坐标数组。</param>
        /// <param name="z">Z值数组。</param>
        /// <returns>变异函数参数。</returns>
        private double[] InitializeVariogramModel(string model, double[] parameters, double[] x, double[] y, double[] z)
        {
            string[] validModels = { "linear", "spherical", "exponential", "gaussian", "power" };
            if (!validModels.Contains(model))
                throw new ArgumentException("不支持的变异函数模型: " + model);

            if (parameters != null && parameters.Length > 0)
                return parameters;

            var (lags, semivariance) = CalculateExperimentalVariogram(x, y, z);
            return FitVariogramModel(model, lags, semivariance);
        }

        /// <summary>
        /// 计算实验变异函数。
        /// </summary>
        /// <param name="x">X坐标数组。</param>
        /// <param name="y">Y坐标数组。</param>
        /// <param name="z">Z值数组。</param>
        /// <returns>滞后距离和半变异函数值。</returns>
        private (double[] lags, double[] semivariance) CalculateExperimentalVariogram(double[] x, double[] y, double[] z)
        {
            int numPoints = x.Length;
            List<double> distances = new List<double>();
            List<double> variances = new List<double>();

            for (int i = 0; i < numPoints; i++)
            {
                for (int j = i + 1; j < numPoints; j++)
                {
                    double dx = x[i] - x[j];
                    double dy = y[i] - y[j];
                    double distance = Math.Sqrt(dx * dx + dy * dy);
                    double variance = 0.5 * Math.Pow(z[i] - z[j], 2);

                    distances.Add(distance);
                    variances.Add(variance);
                }
            }

            int numBins = 20;
            double maxDistance = distances.Max();
            double binSize = maxDistance / numBins;

            double[] lags = new double[numBins];
            double[] semivariance = new double[numBins];
            int[] counts = new int[numBins];

            for (int i = 0; i < distances.Count; i++)
            {
                int binIndex = Math.Min((int)(distances[i] / binSize), numBins - 1);
                lags[binIndex] += distances[i];
                semivariance[binIndex] += variances[i];
                counts[binIndex]++;
            }

            for (int i = 0; i < numBins; i++)
            {
                if (counts[i] > 0)
                {
                    lags[i] /= counts[i];
                    semivariance[i] /= counts[i];
                }
            }

            return (lags.Where((v, i) => counts[i] > 0).ToArray(),
                    semivariance.Where((v, i) => counts[i] > 0).ToArray());
        }

        /// <summary>
        /// 拟合变异函数模型。
        /// </summary>
        /// <param name="model">变异函数模型。</param>
        /// <param name="lags">滞后距离。</param>
        /// <param name="semivariance">半变异函数值。</param>
        /// <returns>变异函数参数。</returns>
        private double[] FitVariogramModel(string model, double[] lags, double[] semivariance)
        {
            Func<double, double[], double> modelFunc = (h, p) => model switch
            {
                "spherical" => p[2] + (h >= p[1] ? p[0] : p[0] * (1.5 * h / p[1] - 0.5 * Math.Pow(h / p[1], 3))),
                "exponential" => p[2] + p[0] * (1 - Math.Exp(-3 * h / p[1])),
                "gaussian" => p[2] + p[0] * (1 - Math.Exp(-3 * Math.Pow(h / p[1], 2))),
                "linear" => p[0] * h + p[1],
                "power" => p[2] + p[0] * Math.Pow(h, p[1]),
                _ => throw new ArgumentException("无效的模型")
            };

            double[] initialParams = model switch
            {
                "spherical" => new[] { semivariance.Max(), lags.Max() / 2, semivariance.Min() },
                "exponential" => new[] { semivariance.Max(), lags.Max() / 2, semivariance.Min() },
                "gaussian" => new[] { semivariance.Max(), lags.Max() / 2, semivariance.Min() },
                "linear" => new[] { semivariance.Max() / lags.Max(), 0.0 },
                "power" => new[] { 1.0, 1.0, 0.0 },
                _ => throw new ArgumentException("无效的模型")
            };

            NelderMead nm = new NelderMead();
            return nm.Optimize((p) =>
            {
                double sum = 0;
                for (int i = 0; i < lags.Length; i++)
                {
                    double modeled = modelFunc(lags[i], p);
                    sum += Math.Pow(modeled - semivariance[i], 2);
                }
                return sum;
            }, initialParams, maxIterations: 1000);
        }

        /// <summary>
        /// 执行克里金插值。
        /// </summary>
        /// <param name="style">插值风格（目前仅支持 "points"）。</param>
        /// <param name="xPoints">预测点的X坐标。</param>
        /// <param name="yPoints">预测点的Y坐标。</param>
        /// <returns>预测值。</returns>
        public double[] Execute(string style, double[] xPoints, double[] yPoints)
        {
            if (style != "points")
                throw new NotSupportedException("仅支持 'points' 风格");

            int n = xAdjusted.Length;
            int numPredictionPoints = xPoints.Length;
            double[] predictions = new double[numPredictionPoints];

            double[,] krigingMatrix = BuildKrigingMatrix(n);

            for (int p = 0; p < numPredictionPoints; p++)
            {
                double[] rhs = new double[n + 1];
                for (int i = 0; i < n; i++)
                {
                    double dx = xPoints[p] - xAdjusted[i];
                    double dy = yPoints[p] - yAdjusted[i];
                    double distance = Math.Sqrt(dx * dx + dy * dy);
                    rhs[i] = c0 - CalculateVariogramValue(distance);
                }
                rhs[n] = 1.0;

                double[] weights = SolveLinearSystem(krigingMatrix, rhs);
                predictions[p] = weights.Take(n).Select((w, i) => w * zValues[i]).Sum();
            }

            return predictions;
        }

        /// <summary>
        /// 构建克里金矩阵。
        /// </summary>
        /// <param name="n">点数。</param>
        /// <returns>克里金矩阵。</returns>
        private double[,] BuildKrigingMatrix(int n)
        {
            double[,] matrix = new double[n + 1, n + 1];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double dx = xAdjusted[i] - xAdjusted[j];
                    double dy = yAdjusted[i] - yAdjusted[j];
                    double distance = Math.Sqrt(dx * dx + dy * dy);
                    matrix[i, j] = c0 - CalculateVariogramValue(distance);
                }
                matrix[i, n] = 1.0;
                matrix[n, i] = 1.0;
            }
            matrix[n, n] = 0.0;

            return matrix;
        }

        /// <summary>
        /// 计算变异函数值。
        /// </summary>
        /// <param name="distance">距离。</param>
        /// <returns>变异函数值。</returns>
        private double CalculateVariogramValue(double distance)
        {
            return variogramModel switch
            {
                "spherical" => SphericalVariogram(distance),
                "exponential" => ExponentialVariogram(distance),
                "gaussian" => GaussianVariogram(distance),
                "linear" => LinearVariogram(distance),
                "power" => PowerVariogram(distance),
                _ => throw new InvalidOperationException("无效的变异函数模型")
            };
        }

        /// <summary>
        /// 球状变异函数。
        /// </summary>
        /// <param name="h">距离。</param>
        /// <returns>变异函数值。</returns>
        private double SphericalVariogram(double h)
        {
            double sill = variogramModelParameters[0];
            double range = variogramModelParameters[1];
            double nugget = variogramModelParameters[2];
            return h >= range ? sill + nugget : nugget + sill * (1.5 * h / range - 0.5 * Math.Pow(h / range, 3));
        }

        /// <summary>
        /// 指数变异函数。
        /// </summary>
        /// <param name="h">距离。</param>
        /// <returns>变异函数值。</returns>
        private double ExponentialVariogram(double h)
        {
            double sill = variogramModelParameters[0];
            double range = variogramModelParameters[1];
            double nugget = variogramModelParameters[2];
            return nugget + sill * (1 - Math.Exp(-3 * h / range));
        }

        /// <summary>
        /// 高斯变异函数。
        /// </summary>
        /// <param name="h">距离。</param>
        /// <returns>变异函数值。</returns>
        private double GaussianVariogram(double h)
        {
            double sill = variogramModelParameters[0];
            double range = variogramModelParameters[1];
            double nugget = variogramModelParameters[2];
            return nugget + sill * (1 - Math.Exp(-3 * Math.Pow(h / range, 2)));
        }

        /// <summary>
        /// 线性变异函数。
        /// </summary>
        /// <param name="h">距离。</param>
        /// <returns>变异函数值。</returns>
        private double LinearVariogram(double h)
        {
            double slope = variogramModelParameters[0];
            double nugget = variogramModelParameters[1];
            return nugget + slope * h;
        }

        /// <summary>
        /// 幂函数变异函数。
        /// </summary>
        /// <param name="h">距离。</param>
        /// <returns>变异函数值。</returns>
        private double PowerVariogram(double h)
        {
            double scale = variogramModelParameters[0];
            double exponent = variogramModelParameters[1];
            double nugget = variogramModelParameters[2];
            return nugget + scale * Math.Pow(h, exponent);
        }

        /// <summary>
        /// 解线性方程组。
        /// </summary>
        /// <param name="matrix">系数矩阵。</param>
        /// <param name="rhs">右侧向量。</param>
        /// <returns>解向量。</returns>
        private double[] SolveLinearSystem(double[,] matrix, double[] rhs)
        {
            int n = rhs.Length;
            double[,] a = (double[,])matrix.Clone();
            double[] b = (double[])rhs.Clone();

            for (int i = 0; i < n; i++)
            {
                int maxRow = i;
                for (int j = i + 1; j < n; j++)
                    if (Math.Abs(a[j, i]) > Math.Abs(a[maxRow, i]))
                        maxRow = j;

                for (int j = i; j < n; j++)
                    (a[i, j], a[maxRow, j]) = (a[maxRow, j], a[i, j]);
                (b[i], b[maxRow]) = (b[maxRow], b[i]);

                for (int j = i + 1; j < n; j++)
                {
                    double factor = a[j, i] / a[i, i];
                    for (int k = i; k < n; k++)
                        a[j, k] -= factor * a[i, k];
                    b[j] -= factor * b[i];
                }
            }

            double[] x = new double[n];
            for (int i = n - 1; i >= 0; i--)
            {
                x[i] = b[i];
                for (int j = i + 1; j < n; j++)
                    x[i] -= a[i, j] * x[j];
                x[i] /= a[i, i];
            }

            return x;
        }

        /// <summary>
        /// 内尔德-米德优化算法类。
        /// </summary>
        private class NelderMead
        {
            /// <summary>
            /// 优化目标函数。
            /// </summary>
            /// <param name="func">目标函数。</param>
            /// <param name="initial">初始点。</param>
            /// <param name="maxIterations">最大迭代次数（默认为 1000）。</param>
            /// <returns>优化后的参数。</returns>
            public double[] Optimize(Func<double[], double> func, double[] initial, int maxIterations = 1000)
            {
                int n = initial.Length;
                List<double[]> simplex = new List<double[]> { initial };
                for (int i = 0; i < n; i++)
                {
                    double[] point = (double[])initial.Clone();
                    point[i] += initial[i] == 0 ? 0.1 : initial[i] * 0.1;
                    simplex.Add(point);
                }

                for (int iter = 0; iter < maxIterations; iter++)
                {
                    simplex.Sort((a, b) => func(a).CompareTo(func(b)));
                    double[] best = simplex[0];
                    double[] worst = simplex[n];
                    double[] centroid = CalculateCentroid(simplex.Take(n).ToList());

                    double[] reflected = Reflect(worst, centroid, 1.0);
                    if (func(reflected) < func(simplex[n - 1]))
                        simplex[n] = reflected;
                    else
                    {
                        if (func(reflected) < func(worst))
                            simplex[n] = reflected;
                        double[] contracted = Contract(worst, centroid);
                        if (func(contracted) < func(worst))
                            simplex[n] = contracted;
                        else
                            Shrink(simplex, best);
                    }
                }

                simplex.Sort((a, b) => func(a).CompareTo(func(b)));
                return simplex[0];
            }

            /// <summary>
            /// 计算重心。
            /// </summary>
            /// <param name="simplex">单纯形。</param>
            /// <returns>重心。</returns>
            private double[] CalculateCentroid(List<double[]> simplex)
            {
                int n = simplex[0].Length;
                double[] centroid = new double[n];
                foreach (double[] point in simplex)
                    for (int i = 0; i < n; i++)
                        centroid[i] += point[i];
                for (int i = 0; i < n; i++)
                    centroid[i] /= simplex.Count;
                return centroid;
            }

            /// <summary>
            /// 反射操作。
            /// </summary>
            /// <param name="point">点。</param>
            /// <param name="centroid">重心。</param>
            /// <param name="alpha">反射系数。</param>
            /// <returns>反射后的点。</returns>
            private double[] Reflect(double[] point, double[] centroid, double alpha)
            {
                double[] reflected = new double[point.Length];
                for (int i = 0; i < point.Length; i++)
                    reflected[i] = centroid[i] + alpha * (centroid[i] - point[i]);
                return reflected;
            }

            /// <summary>
            /// 收缩操作。
            /// </summary>
            /// <param name="point">点。</param>
            /// <param name="centroid">重心。</param>
            /// <returns>收缩后的点。</returns>
            private double[] Contract(double[] point, double[] centroid)
            {
                return Reflect(point, centroid, -0.5);
            }

            /// <summary>
            /// 收缩单纯形。
            /// </summary>
            /// <param name="simplex">单纯形。</param>
            /// <param name="best">最佳点。</param>
            private void Shrink(List<double[]> simplex, double[] best)
            {
                for (int i = 1; i < simplex.Count; i++)
                    for (int j = 0; j < simplex[i].Length; j++)
                        simplex[i][j] = best[j] + 0.5 * (simplex[i][j] - best[j]);
            }
        }
    }
}