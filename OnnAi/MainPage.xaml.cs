using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Windows.AI.MachineLearning;
using Windows.Graphics.Imaging;
using Windows.Media;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.Storage.Streams;
using Windows.UI.Core;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Media.Imaging;

namespace OnnxAi
{
    /// <summary>
    /// 
    /// </summary>
    public sealed partial class MainPage : Page
    {
        private Stopwatch _stopwatch = new Stopwatch();
        private OnnxModel _model = null;
        //private const string _onnxFileName = "ArtModel_R4.onnx";
        private const string _onnxFileName = "ArtModel_R6.onnx";


        public sealed class OnnxModelInput
        {
            public VideoFrame data { get; set; }
        }

        public sealed class OnnxModelOutput
        {
            public IReadOnlyList<string> classLabel { get; set; }
            public IDictionary<string, float> loss { get; set; }
        }

        public sealed class OnnxModel
        {
            private LearningModel learningModel;
            private LearningModelSession session;
            private LearningModelBinding binding;

            public static async Task<OnnxModel> CreateFromStreamAsync(IRandomAccessStreamReference stream)
            {
                OnnxModel onnxModel = new OnnxModel();
                onnxModel.learningModel = await LearningModel.LoadFromStreamAsync(stream);
                onnxModel.session = new LearningModelSession(onnxModel.learningModel);
                onnxModel.binding = new LearningModelBinding(onnxModel.session);
                return onnxModel;
            }

            public async Task<OnnxModelOutput> EvaluateAsync(OnnxModelInput input)
            {
                binding.Bind("data", input.data);
                var result = await session.EvaluateAsync(binding, string.Empty);

                OnnxModelOutput output = new OnnxModelOutput();
                output.classLabel = (result.Outputs["classLabel"] as TensorString).GetAsVectorView();
                var predictions = result.Outputs["loss"] as IList<IDictionary<string, float>>;
                output.loss = predictions[0];

                return output;
            }
        }

        public MainPage()
        {
            this.InitializeComponent();
        }

        private async Task LoadModelAsync()
        {
            await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"Caricamento di {_onnxFileName} in corso, attendere...");

            try
            {
                _stopwatch = Stopwatch.StartNew();

                var modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/{_onnxFileName}"));
                _model = await OnnxModel.CreateFromStreamAsync(modelFile);

                _stopwatch.Stop();
                Debug.WriteLine($"{_onnxFileName} caricato. Tempo trascorso: {_stopwatch.ElapsedMilliseconds} ms");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"error: {ex.Message}");
                await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"error: {ex.Message}");
                _model = null;
            }
        }

        private async void ButtonRun_Click(object sender, RoutedEventArgs e)
        {
            ButtonRun.IsEnabled = false;
            UIPreviewImage.Source = null;
            try
            {
                if (_model == null)
                {
                    // Load the model
                    await Task.Run(async () => await LoadModelAsync());
                }

                // Trigger file picker to select an image file
                FileOpenPicker fileOpenPicker = new FileOpenPicker();
                fileOpenPicker.SuggestedStartLocation = PickerLocationId.PicturesLibrary;
                fileOpenPicker.FileTypeFilter.Add(".bmp");
                fileOpenPicker.FileTypeFilter.Add(".jpeg");
                fileOpenPicker.FileTypeFilter.Add(".jpg");
                fileOpenPicker.FileTypeFilter.Add(".png");
                fileOpenPicker.ViewMode = PickerViewMode.Thumbnail;
                StorageFile selectedStorageFile = await fileOpenPicker.PickSingleFileAsync();

                SoftwareBitmap softwareBitmap;
                using (IRandomAccessStream stream = await selectedStorageFile.OpenAsync(FileAccessMode.Read))
                {
                    // Create the decoder from the stream 
                    BitmapDecoder decoder = await BitmapDecoder.CreateAsync(stream);

                    // Get the SoftwareBitmap representation of the file in BGRA8 format
                    softwareBitmap = await decoder.GetSoftwareBitmapAsync();
                    softwareBitmap = SoftwareBitmap.Convert(softwareBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
                }

                // Display the image
                SoftwareBitmapSource imageSource = new SoftwareBitmapSource();
                await imageSource.SetBitmapAsync(softwareBitmap);
                UIPreviewImage.Source = imageSource;

                // Encapsulate the image in the WinML image type (VideoFrame) to be bound and evaluated
                VideoFrame inputImage = VideoFrame.CreateWithSoftwareBitmap(softwareBitmap);

                await Task.Run(async () =>
                {
                    // Evaluate the image
                    await EvaluateVideoFrameAsync(inputImage);
                });
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"error: {ex.Message}");
                await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"error: {ex.Message}");
            }
            finally
            {
                ButtonRun.IsEnabled = true;
            }
        }

        private async Task EvaluateVideoFrameAsync(VideoFrame frame)
        {
            if (frame != null)
            {
                try
                {
                    _stopwatch.Restart();
                    OnnxModelInput inputData = new OnnxModelInput();
                    inputData.data = frame;
                    var results = await _model.EvaluateAsync(inputData);
                    var loss = results.loss.ToList().OrderBy(x=>-(x.Value));
                    var labels = results.classLabel;
                    _stopwatch.Stop();

                    var lossStr = string.Join(",  ", loss.Select(l => l.Key + " " + (l.Value * 100.0f).ToString("#0.00") + "%"));
                    string message = $"Tempo impiegato per la valutazione {_stopwatch.ElapsedMilliseconds}ms ,\nPredizioni: {lossStr}.";
                    Debug.WriteLine(message);
                    await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = message);
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"error: {ex.Message}");
                    await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => StatusBlock.Text = $"error: {ex.Message}");
                }

                await Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () => ButtonRun.IsEnabled = true);
            }
        }
    }
}
