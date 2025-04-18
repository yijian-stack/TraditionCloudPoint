using System;

namespace PointCloudProcessorNamespace
{
    /// <summary>
    /// 程序入口类，用于调用点云处理逻辑。
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            var processor = new PointCloudProcessorNamespace.PointCloudProcessor();
            var parameters = new PointCloudProcessorNamespace.PointCloudProcessor.Params();

            // 输入路径
            var inputPath = @"D:\company\0125\u3dbudian\input\1-1.csv";
            var outputPath = $@"D:\company\0125\u3dbudian\input\1-1a.csv";
            // 自定义范围参数（可选）
            parameters.XMin = 53120;
            parameters.XMax = 161315;
            parameters.YMin = 2116;
            parameters.YMax = 457851;

            // 是否自动获取范围（0：否，手动设置范围；1：是，根据数据本身范围）
            parameters.IsRangeAuto = 1;

            // 调用处理方法
            processor.ProcessPointCloud(inputPath, outputPath, parameters);

        }
    }
}