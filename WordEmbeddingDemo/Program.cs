using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Transforms.Text;

namespace WordEmbeddingDemo
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();
            var emptyData = context.Data.LoadFromEnumerable(new List<TextInput>());

            var pipeline = context.Transforms.Text.NormalizeText("Text", null, keepDiacritics: false, keepPunctuations: false, keepNumbers: false)
                .Append(context.Transforms.Text.TokenizeIntoWords("Tokens", "Text"))
                .Append(context.Transforms.Text.ApplyWordEmbedding("Features", "Tokens", WordEmbeddingEstimator.PretrainedModelKind.SentimentSpecificWordEmbedding));

            var transformer = pipeline.Fit(emptyData);
            var predictionEngine = context.Model.CreatePredictionEngine<TextInput, TextFeatures>(transformer);

            var wordOfEmbedding = new TextInput { Text = "example" };
            var prediction = predictionEngine.Predict(wordOfEmbedding);

            Console.WriteLine("Merkmalesvektor: ");
            foreach (var feature in prediction.Features)
            {
                Console.Write($"{feature:F4} ");
            }

            Console.ReadLine();
            //Console.WriteLine(Environment.NewLine);

            //var wordOfEmbeddingTwo = new TextInput { Text = "Console" };
            //var predictionTwo = predictionEngine.Predict(wordOfEmbeddingTwo);

            //Console.WriteLine("Merkmalsvektor: ");
            //foreach (var feature in predictionTwo.Features)
            //{
            //    Console.Write($"{feature:F4} ");
            //}

            //Console.WriteLine(Environment.NewLine);

            //var wordOfEmbeddingThree = new TextInput { Text = "Garage door" };
            //var predictionThree = predictionEngine.Predict(wordOfEmbeddingThree);

            //Console.WriteLine("Merkmalsvektor: ");
            //foreach (var feature in predictionThree.Features)
            //{
            //    Console.Write($"{feature:F4} ");
            //}

            //Console.ReadLine();
        }
    }

    public class TextInput
    {
        public string Text { get; set; }
    }

    public class TextFeatures
    {
        public float[] Features { get; set; }
    }
}
