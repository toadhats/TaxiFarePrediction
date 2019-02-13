using System;
using System.Globalization;
using System.IO;
using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Normalizers;
using Microsoft.ML.Transforms.Text;

namespace TaxiFarePrediction
{
    class Program
    {
        // TODO: What's the best way to get these into cloud storage so this application could be deployed as a lambda?
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        static TextLoader _textLoader;

        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 0);

            _textLoader = mlContext.Data.CreateTextLoader(new TextLoader.Arguments()
            {
                Separators = new[] { ',' },
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("VendorId", DataKind.Text, 0),
                    new TextLoader.Column("RateCode", DataKind.Text, 1), // Peak, offpeak?
                    new TextLoader.Column("PassengerCount", DataKind.R4, 2),
                    new TextLoader.Column("TripTime", DataKind.R4, 3), // Duration in seconds
                    new TextLoader.Column("TripDistance", DataKind.R4, 4), // Decimal miles
                    new TextLoader.Column("PaymentType", DataKind.Text, 5),
                    new TextLoader.Column("FareAmount", DataKind.R4, 6) // Decimal dollars
                }
            });
            // Don't regenerate and reevaluate the model every time, takes too long
            // TODO: Make it regenerate the model if the training data has changed? (Would need to persist a hash of the training data between runs)
            if (!File.Exists(_modelPath) )
            {
                var model = Train(mlContext, _trainDataPath); // Generate model from training dataset - TODO: No reason to do this every time!
                Evaluate(mlContext, model); // Evaluate model performance against the test data (TODO: Parameterise to allow evaluation against different datasets without modifying source)
            }
            //TestSinglePrediction(mlContext);

            // Let's actually use this as if it was a real service

            // TODO: Make it possible to specify this trip from command line
            var culture = new CultureInfo("en-US");
            var trip = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = int.Parse(args[0], culture),//1,
                TripTime = int.Parse(args[1], culture),//1540,
                TripDistance = float.Parse(args[2], culture), // 5.70f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict.
            };
            var result = Predict(mlContext, trip);

            Console.WriteLine($"Predicted fare: ${result.FareAmount:#.##}");

        }
        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = _textLoader.Read(dataPath);

            var pipeline = mlContext.Transforms.CopyColumns(inputColumnName: "FareAmount", outputColumnName: "Label")
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorId"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCode"))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding("PaymentType"))
                .Append(mlContext.Transforms.Concatenate("Features", "VendorId", "RateCode", "PassengerCount", "TripTime", "TripDistance", "PaymentType"))
                .Append(mlContext.Regression.Trainers.FastTree());

            var model = pipeline.Fit(dataView);
            SaveModelAsFile(mlContext, model);
            return model;

        }
        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = _textLoader.Read(_testDataPath);
            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score"); // Compute metrics via regression evaluators

            // TODO: This could be done more elegantly - I prefer something like a function PrintBanner(string s)
            Console.WriteLine();
            Console.WriteLine($"*************************************************");
            Console.WriteLine($"*       Model quality metrics evaluation         ");
            Console.WriteLine($"*------------------------------------------------");

            Console.WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}"); // R^2 = coefficient of determination
            Console.WriteLine($"*       RMS loss:      {metrics.Rms:#.##}"); // Root mean square loss
        }

        /// <summary>
        /// Extending this with something actually useful
        /// </summary>
        /// <param name="mLContext">Pass context in because I'm not a fan of state</param>
        /// <param name="trip">Trip details for which to predict price</param>
        /// <returns></returns>
        private static TaxiTripFarePrediction Predict(MLContext mlContext, TaxiTrip trip ) // This should not really be in Program.cs, make public and move this and everything else out into an Engine class?
        {
            // TODO: Cache model? Going from disk every time seems extremely inefficient.
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            var predictionFunction = loadedModel.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(mlContext);

            var prediction = predictionFunction.Predict(trip);
            return prediction;
        }

        private static void TestSinglePrediction(MLContext mlContext)
        {
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            // PredictionEngine is a "wrapper" around a model that allows predictions on individual examples (most common scenario in production for consumer-scale applications? Or is batching more common?
            var predictionFunction = loadedModel.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(mlContext);

            // TODO: Parameterise
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 1,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };

            var prediction = predictionFunction.Predict(taxiTripSample);

            //TODO: PrintBanner again
            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            Console.WriteLine($"**********************************************************************");
        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fileStream);

            Console.WriteLine("Model saved to {0}", _modelPath);
        }
    }
}
