using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using kriging.PointCloudInterpolation;

namespace PointCloudProcessorNamespace
{
    /// <summary>
    /// 点云处理类，负责对点云数据进行插值、筛选和网格划分等操作。
    /// </summary>
    public class PointCloudProcessor
    {
        /// <summary>
        /// 点云中的单个点，包含X、Y、Z坐标。
        /// </summary>
        public class Point
        {
            /// <summary>
            /// X坐标。
            /// </summary>
            public double X { get; set; }

            /// <summary>
            /// Y坐标。
            /// </summary>
            public double Y { get; set; }

            /// <summary>
            /// Z坐标。
            /// </summary>
            public double Z { get; set; }
        }

        /// <summary>
        /// 点云处理参数类，用于配置插值和筛选的参数。
        /// </summary>
        public class Params
        {
            /// <summary>
            /// 最大网格数量。
            /// </summary>
            public int MaximumGrids { get; set; } = 3000;

            /// <summary>
            /// 密度阈值（每平方公里的点数）。
            /// </summary>
            public double DensityThreshold { get; set; } = 4.0;

            /// <summary>
            /// 噪声比例（用于插值时的噪声添加）。
            /// </summary>
            public double NoiseScale { get; set; } = 0.001;

            /// <summary>
            /// 最大采样点数。
            /// </summary>
            public int MaximumSamples { get; set; } = 50;

            /// <summary>
            /// 最小邻居点数。
            /// </summary>
            public int MinimumNeighborPoints { get; set; } = 10;

            /// <summary>
            /// 子网格划分数量。
            /// </summary>
            public int Subdivision { get; set; } = 5;

            /// <summary>
            /// 协方差模型（支持 "spherical"、"exponential" 等）。
            /// </summary>
            public string VariogramModel { get; set; } = "spherical";

            /// <summary>
            /// 是否自动计算范围（0：否，手动设置范围；1：是，根据数据范围）。
            /// </summary>
            public double IsRangeAuto { get; set; } = 0;

            /// <summary>
            /// X坐标范围（最小值）。
            /// </summary>
            public double XMin { get; set; } = 53120;

            /// <summary>
            /// X坐标范围（最大值）。
            /// </summary>
            public double XMax { get; set; } = 161315;

            /// <summary>
            /// Y坐标范围（最小值）。
            /// </summary>
            public double YMin { get; set; } = 2116;

            /// <summary>
            /// Y坐标范围（最大值）。
            /// </summary>
            public double YMax { get; set; } = 457851;
        }

        /// <summary>
        /// 处理点云数据，包括插值、筛选和网格划分。
        /// </summary>
        /// <param name="inputPath">输入文件路径。</param>
        /// <param name="outputPath">输出文件路径。</param>
        /// <param name="parameters">处理参数（可选）。</param>
        public void ProcessPointCloud(string inputPath, string outputPath, Params parameters = null)
        {
            if (parameters == null)
            {
                parameters = new Params();
            }

            // 自动计算范围（如果启用）
            if (parameters.IsRangeAuto == 1)
            {
                var (xAll, yAll, zAll) = ReadCsv(inputPath);
                parameters.XMin = xAll.Min();
                parameters.XMax = xAll.Max();
                parameters.YMin = yAll.Min();
                parameters.YMax = yAll.Max();
            }


            var startTime = DateTime.Now;

            // 创建临时文件
            var tempOutput = outputPath + ".temp";
            File.Copy(inputPath, tempOutput, true);

            // 三次补点操作
            for (int iteration = 0; iteration < 3; iteration++)
            {

                // 更新范围的左下角坐标
                parameters.XMin = parameters.XMin + iteration * 500;
                parameters.YMin = parameters.YMin + iteration * 500;

                var readTime = DateTime.Now;
                var (xAll, yAll, zAll) = ReadCsv(inputPath);

                var filterTime = DateTime.Now;
                var filtered = FilterPoints(xAll, yAll, zAll, parameters);

                var gridTime = DateTime.Now;
                var (xGrids, yGrids, xGridSize, yGridSize, xMin, yMin) = CreateGrid(filtered.x, filtered.y);
                var gridIndex = AssignPointsToGrid(filtered, xGrids, yGrids, xGridSize, yGridSize, xMin, yMin);

                var interpolationTime = DateTime.Now;
                var allRefinedPoints = ProcessGridInterpolation(
                    gridIndex, xGrids, yGrids, xGridSize, yGridSize, xMin, yMin, parameters);

                AppendNewPoints(tempOutput, allRefinedPoints);
            }

            // 将临时文件重命名为最终文件
            File.Move(tempOutput, outputPath, true);

        }

        /// <summary>
        /// 从CSV文件读取点云数据。
        /// </summary>
        /// <param name="filePath">CSV文件路径。</param>
        /// <returns>包含X、Y、Z坐标的元组。</returns>
        private static (double[] x, double[] y, double[] z) ReadCsv(string filePath)
        {
            var lines = File.ReadAllLines(filePath);
            var x = new double[lines.Length];
            var y = new double[lines.Length];
            var z = new double[lines.Length];

            Parallel.For(0, lines.Length, i =>
            {
                var parts = lines[i].Split(',');
                x[i] = double.Parse(parts[0]);
                y[i] = double.Parse(parts[1]);
                z[i] = double.Parse(parts[2]);
            });

            return (x, y, z);
        }

        /// <summary>
        /// 筛选范围内的点。
        /// </summary>
        /// <param name="xAll">所有X坐标。</param>
        /// <param name="yAll">所有Y坐标。</param>
        /// <param name="zAll">所有Z坐标。</param>
        /// <param name="parameters">筛选参数。</param>
        /// <returns>筛选后的点。</returns>
        private static (double[] x, double[] y, double[] z) FilterPoints(
            double[] xAll, double[] yAll, double[] zAll, Params parameters)
        {
            var filtered = new List<(double x, double y, double z)>();
            for (int i = 0; i < xAll.Length; i++)
            {
                if (xAll[i] >= parameters.XMin && xAll[i] <= parameters.XMax &&
                    yAll[i] >= parameters.YMin && yAll[i] <= parameters.YMax)
                {
                    filtered.Add((xAll[i], yAll[i], zAll[i]));
                }
            }
            return (
                filtered.Select(p => p.x).ToArray(),
                filtered.Select(p => p.y).ToArray(),
                filtered.Select(p => p.z).ToArray()
            );
        }

        /// <summary>
        /// 创建网格。
        /// </summary>
        /// <param name="x">X坐标数组。</param>
        /// <param name="y">Y坐标数组。</param>
        /// <returns>网格参数。</returns>
        private static (int xGrids, int yGrids, double xSize, double ySize, double xMin, double yMin)
            CreateGrid(double[] x, double[] y)
        {
            var xMin = x.Min();
            var xMax = x.Max();
            var yMin = y.Min();
            var yMax = y.Max();

            var xRange = xMax - xMin;
            var yRange = yMax - yMin;
            var aspectRatio = xRange / yRange;

            var yGrids = (int)Math.Sqrt(3000 / aspectRatio);
            var xGrids = (int)(aspectRatio * yGrids);

            return (xGrids, yGrids, xRange / xGrids, yRange / yGrids, xMin, yMin);
        }

        /// <summary>
        /// 计算网格密度。
        /// </summary>
        /// <param name="pointCount">点数。</param>
        /// <param name="xSize">网格X尺寸。</param>
        /// <param name="ySize">网格Y尺寸。</param>
        /// <returns>密度值。</returns>
        private static double CalculateDensity(int pointCount, double xSize, double ySize)
        {
            var area = (xSize * ySize) / 1e6;
            return pointCount / area;
        }

        /// <summary>
        /// 初始化网格。
        /// </summary>
        /// <param name="xGrids">X方向网格数。</param>
        /// <param name="yGrids">Y方向网格数。</param>
        /// <returns>网格数组。</returns>
        private static List<Point>[][] InitGrid(int xGrids, int yGrids)
        {
            var grid = new List<Point>[xGrids][];
            for (int i = 0; i < xGrids; i++)
            {
                grid[i] = new List<Point>[yGrids];
                for (int j = 0; j < yGrids; j++)
                {
                    grid[i][j] = new List<Point>();
                }
            }
            return grid;
        }

        /// <summary>
        /// 将点分配到网格中。
        /// </summary>
        /// <param name="points">点云数据。</param>
        /// <param name="xGrids">X方向网格数。</param>
        /// <param name="yGrids">Y方向网格数。</param>
        /// <param name="xGridSize">X方向网格尺寸。</param>
        /// <param name="yGridSize">Y方向网格尺寸。</param>
        /// <param name="xMin">X最小值。</param>
        /// <param name="yMin">Y最小值。</param>
        /// <returns>网格数组。</returns>
        private static List<Point>[][] AssignPointsToGrid(
            (double[] x, double[] y, double[] z) points,
            int xGrids, int yGrids,
            double xGridSize, double yGridSize,
            double xMin, double yMin)
        {
            var grid = InitGrid(xGrids, yGrids);
            Parallel.For(0, points.x.Length, i =>
            {
                var xi = (int)((points.x[i] - xMin) / xGridSize);
                var yi = (int)((points.y[i] - yMin) / yGridSize);
                if (xi >= 0 && xi < xGrids && yi >= 0 && yi < yGrids)
                {
                    lock (grid[xi][yi])
                    {
                        grid[xi][yi].Add(new Point
                        {
                            X = points.x[i],
                            Y = points.y[i],
                            Z = points.z[i]
                        });
                    }
                }
            });
            return grid;
        }

        /// <summary>
        /// 追加新插值点到文件。
        /// </summary>
        /// <param name="path">文件路径。</param>
        /// <param name="points">新插值点。</param>
        private static void AppendNewPoints(string path, List<Point> points)
        {
            using var writer = new StreamWriter(path, true);
            foreach (var p in points)
            {
                writer.WriteLine($"{p.X},{p.Y},{p.Z}");
            }
        }

        /// <summary>
        /// 获取子网格密度。
        /// </summary>
        /// <param name="points">点云数据。</param>
        /// <param name="gridI">网格X索引。</param>
        /// <param name="gridJ">网格Y索引。</param>
        /// <param name="xSize">网格X尺寸。</param>
        /// <param name="ySize">网格Y尺寸。</param>
        /// <param name="xMin">X最小值。</param>
        /// <param name="yMin">Y最小值。</param>
        /// <param name="parameters">处理参数。</param>
        /// <returns>子网格密度数组。</returns>
        private static double[] GetSubGridDensities(
            List<Point> points,
            int gridI, int gridJ,
            double xSize, double ySize,
            double xMin, double yMin,
            Params parameters)
        {
            var subSizeX = xSize / parameters.Subdivision;
            var subSizeY = ySize / parameters.Subdivision;
            var gridXStart = xMin + gridI * xSize;
            var gridYStart = yMin + gridJ * ySize;

            var counts = new int[parameters.Subdivision, parameters.Subdivision];

            foreach (var p in points)
            {
                var subI = (int)((p.X - gridXStart) / subSizeX);
                var subJ = (int)((p.Y - gridYStart) / subSizeY);
                if (subI >= 0 && subI < parameters.Subdivision &&
                    subJ >= 0 && subJ < parameters.Subdivision)
                {
                    counts[subI, subJ]++;
                }
            }

            var subArea = (subSizeX * subSizeY) / 1e6;
            return counts.Cast<int>().Select(c => c / subArea).ToArray();
        }

        /// <summary>
        /// 执行网格插值。
        /// </summary>
        /// <param name="grid">网格数组。</param>
        /// <param name="xGrids">X方向网格数。</param>
        /// <param name="yGrids">Y方向网格数。</param>
        /// <param name="xGridSize">X方向网格尺寸。</param>
        /// <param name="yGridSize">Y方向网格尺寸。</param>
        /// <param name="xMin">X最小值。</param>
        /// <param name="yMin">Y最小值。</param>
        /// <param name="parameters">处理参数。</param>
        /// <returns>插值后的点。</returns>
        private static List<Point> ProcessGridInterpolation(
            List<Point>[][] grid,
            int xGrids, int yGrids,
            double xGridSize, double yGridSize,
            double xMin, double yMin,
            Params parameters)
        {
            var allPoints = new List<Point>();
            var random = new Random();

            Parallel.For(0, xGrids, i =>
            {
                var localPoints = new List<Point>();
                for (int j = 0; j < yGrids; j++)
                {
                    var currentPoints = grid[i][j];
                    var density = CalculateDensity(currentPoints.Count, xGridSize, yGridSize);

                    if (density >= parameters.DensityThreshold)
                    {
                        var subDensities = GetSubGridDensities(
                            currentPoints, i, j, xGridSize, yGridSize, xMin, yMin, parameters);
                        if (subDensities.All(d => d >= parameters.DensityThreshold))
                        {
                            continue;
                        }
                    }

                    var sampledPoints = GetSampledPoints(grid, i, j, xGrids, yGrids, parameters);
                    if (sampledPoints.Count < 3) continue;

                    var newPoints = PerformKrigingInterpolation(
                        sampledPoints, i, j, xGridSize, yGridSize, xMin, yMin, random, parameters);

                    lock (allPoints)
                    {
                        localPoints.AddRange(newPoints);
                    }
                }
                lock (allPoints)
                {
                    allPoints.AddRange(localPoints);
                }
            });

            return allPoints;
        }

        /// <summary>
        /// 获取采样点。
        /// </summary>
        /// <param name="grid">网格数组。</param>
        /// <param name="i">网格X索引。</param>
        /// <param name="j">网格Y索引。</param>
        /// <param name="xGrids">X方向网格数。</param>
        /// <param name="yGrids">Y方向网格数。</param>
        /// <param name="parameters">处理参数。</param>
        /// <returns>采样点。</returns>
        private static List<Point> GetSampledPoints(
            List<Point>[][] grid,
            int i, int j,
            int xGrids, int yGrids,
            Params parameters)
        {
            var points = new HashSet<Point>();
            var radius = 1;
            const int maxRadius = 13;
            var neighborPoints = new List<Point>();

            while (neighborPoints.Count < parameters.MinimumNeighborPoints && radius <= maxRadius)
            {
                var iMin = Math.Max(0, i - radius);
                var iMax = Math.Min(xGrids - 1, i + radius);
                var jMin = Math.Max(0, j - radius);
                var jMax = Math.Min(yGrids - 1, j + radius);

                for (int x = iMin; x <= iMax; x++)
                {
                    for (int y = jMin; y <= jMax; y++)
                    {
                        if (x == i && y == j) continue; // 跳过当前网格
                        if (grid[x][y].Any())
                        {
                            if (!neighborPoints.Any(p => p.X == grid[x][y].First().X && p.Y == grid[x][y].First().Y && p.Z == grid[x][y].First().Z))
                            {
                                neighborPoints.Add(grid[x][y].First()); // 如果不存在，则添加到 neighborPoints
                            }

                            if (neighborPoints.Count >= parameters.MinimumNeighborPoints) break;
                        }
                    }
                }
                if (neighborPoints.Count >= parameters.MinimumNeighborPoints) break;
                radius++;
            }

            foreach (var p in neighborPoints)
            {
                points.Add(p);
            }

            if (neighborPoints.Count >= parameters.MinimumNeighborPoints)
            {
                var remaining = parameters.MaximumSamples - neighborPoints.Count;
                if (remaining > 0 && grid[i][j].Count > 0)
                {
                    var currentPoints = grid[i][j].Take(remaining).ToList();

                    foreach (var p in currentPoints)
                    {
                        points.Add(p);
                    }
                }
            }

            return points.ToList();
        }

        /// <summary>
        /// 执行克里金插值。
        /// </summary>
        /// <param name="samplePoints">采样点。</param>
        /// <param name="gridI">网格X索引。</param>
        /// <param name="gridJ">网格Y索引。</param>
        /// <param name="xGridSize">网格X尺寸。</param>
        /// <param name="yGridSize">网格Y尺寸。</param>
        /// <param name="xMin">X最小值。</param>
        /// <param name="yMin">Y最小值。</param>
        /// <param name="random">随机数生成器。</param>
        /// <param name="parameters">处理参数。</param>
        /// <returns>插值后的点。</returns>
        private static List<Point> PerformKrigingInterpolation(
            List<Point> samplePoints,
            int gridI, int gridJ,
            double xGridSize, double yGridSize,
            double xMin, double yMin,
            Random random,
            Params parameters)
        {
            try
            {
                var noisyPoints = samplePoints.Select(p => new Point
                {
                    X = p.X + NextGaussian(random, 0, parameters.NoiseScale),
                    Y = p.Y + NextGaussian(random, 0, parameters.NoiseScale),
                    Z = p.Z
                }).ToList();

                var xSample = noisyPoints.Select(p => p.X).ToArray();
                var ySample = noisyPoints.Select(p => p.Y).ToArray();
                var zSample = noisyPoints.Select(p => p.Z).ToArray();

                var ok = new OrdinaryKriging(xSample, ySample, zSample, parameters.VariogramModel);

                var gridXMin = xMin + gridI * xGridSize;
                var gridYMin = yMin + gridJ * yGridSize;

                var gridArea = (xGridSize * yGridSize) / 1e6;
                var targetPoints = Math.Max(4, (int)(parameters.DensityThreshold * gridArea));
                var n = (int)Math.Ceiling(Math.Sqrt(targetPoints));

                var (predictX, predictY) = GeneratePredictionGrid(
                    gridXMin, gridYMin, xGridSize, yGridSize, n);

                var predictZ = ok.Execute("points", predictX, predictY);

                return ProcessPredictionResults(predictX, predictY, predictZ);
            }
            catch
            {
                return new List<Point>();
            }
        }

        /// <summary>
        /// 生成预测网格。
        /// </summary>
        /// <param name="gridXMin">网格X最小值。</param>
        /// <param name="gridYMin">网格Y最小值。</param>
        /// <param name="xSize">X尺寸。</param>
        /// <param name="ySize">Y尺寸。</param>
        /// <param name="n">网格点数。</param>
        /// <returns>预测网格的X和Y坐标。</returns>
        private static (double[] x, double[] y) GeneratePredictionGrid(
            double gridXMin, double gridYMin,
            double xSize, double ySize,
            int n)
        {
            double xStart = gridXMin + xSize / (2 * n);
            double xEnd = gridXMin + xSize * (1 - 1.0 / (2 * n));
            double yStart = gridYMin + ySize / (2 * n);
            double yEnd = gridYMin + ySize * (1 - 1.0 / (2 * n));

            var xCoords = Linspace(xStart, xEnd, n);
            var yCoords = Linspace(yStart, yEnd, n);

            var predictX = new List<double>();
            var predictY = new List<double>();

            foreach (var x in xCoords)
            {
                foreach (var y in yCoords)
                {
                    predictX.Add(x);
                    predictY.Add(y);
                }
            }

            return (predictX.ToArray(), predictY.ToArray());
        }

        /// <summary>
        /// 生成等间距数组。
        /// </summary>
        /// <param name="start">起始值。</param>
        /// <param name="end">结束值。</param>
        /// <param name="count">元素数量。</param>
        /// <returns>等间距数组。</returns>
        private static double[] Linspace(double start, double end, int count)
        {
            if (count < 2)
                throw new ArgumentException("Count must be at least 2");

            var step = (end - start) / (count - 1);
            var result = new List<double>();

            for (int i = 0; i < count; i++)
            {
                result.Add(start + i * step);
            }

            return result.ToArray();
        }

        /// <summary>
        /// 处理预测结果。
        /// </summary>
        /// <param name="x">X坐标数组。</param>
        /// <param name="y">Y坐标数组。</param>
        /// <param name="z">Z坐标数组。</param>
        /// <returns>处理后的点。</returns>
        private static List<Point> ProcessPredictionResults(double[] x, double[] y, double[] z)
        {
            var points = new List<Point>();
            for (int i = 0; i < x.Length; i++)
            {
                if (!double.IsNaN(z[i]))
                {
                    points.Add(new Point
                    {
                        X = Math.Round(x[i], 2),
                        Y = Math.Round(y[i], 2),
                        Z = Math.Round(z[i], 1)
                    });
                }
            }
            return points;
        }

        /// <summary>
        /// 生成高斯随机数。
        /// </summary>
        /// <param name="rand">随机数生成器。</param>
        /// <param name="mean">均值。</param>
        /// <param name="stdDev">标准差。</param>
        /// <returns>高斯随机数。</returns>
        private static double NextGaussian(Random rand, double mean, double stdDev)
        {
            double u1 = 1.0 - rand.NextDouble();
            double u2 = 1.0 - rand.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                  Math.Sin(2.0 * Math.PI * u2);
            return mean + stdDev * randStdNormal;
        }
    }
}