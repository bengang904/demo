// 载入 YAMNet 类别 CSV
const classMapResp = await fetch("yamnet_class_map.csv");
const classCSV = await classMapResp.text();
const classes = classCSV
  .split("\n")
  .slice(1)
  .map((line) =>
    line
      .split(",")
      .slice(2)
      .join(",")
      .replace(/(\/|"|\\)/g, "")
  );

// 声明全局变量
let audioStream;
let mediaRecorder;
let timer;

// 加载模型
const modelUrl = "https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1";
const model = await tf.loadGraphModel(modelUrl, { fromTFHub: true });

// 获取页面元素
const audio = document.querySelector("audio");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const predict_p = document.getElementById("predict");
const recordBtn = document.getElementById("recordBtn");

// 点击录音按钮
recordBtn.addEventListener("click", async () => {
  try {
    recordBtn.disabled = true;
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

    // 请求麦克风权限
    if (!audioStream) {
      audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    }

    // 设置 MediaRecorder
    const chunks = [];
    mediaRecorder = new MediaRecorder(audioStream);
    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);

    mediaRecorder.onstop = async () => {
      clearInterval(timer);
      predict_p.innerText = "识别中...";

      // 解码音频
      const blob = new Blob(chunks, { type: "audio/webm" });
      const arrayBuffer = await blob.arrayBuffer();
      const audioCtx = new AudioContext();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

      // 播放音频
      audio.src = URL.createObjectURL(blob);
      audio.style.display = "block";

      // 获取音频通道数据并传给模型
      const waveform = tf.tensor(audioBuffer.getChannelData(0));
      const [scores, embeddings, spectrogram] = model.predict(waveform);

      // 获取前10分类
      const top10 = await tf.topk(scores.mean(0), 10).indices.array();
      const top10classes = top10.map((i) => classes[i]).join("\n");
      predict_p.innerText = top10classes;

      // 绘制 spectrogram
      const spectrogram_scaled = await tf
        .transpose(await normalize(spectrogram).NORMALIZED_VALUES, [1, 0])
        .square()
        .reverse(0);

      await tf.browser.toPixels(spectrogram_scaled, canvas);
      recordBtn.disabled = false;
    };

    // 开始录音与倒计时
    mediaRecorder.start();
    let secondsLeft = 3;
    predict_p.innerText = `录音中... ${secondsLeft}s`;
    timer = setInterval(() => {
      secondsLeft--;
      if (secondsLeft > 0) {
        predict_p.innerText = `录音中... ${secondsLeft}s`;
      }
    }, 1000);

    setTimeout(() => mediaRecorder.stop(), 3000);
  } catch (err) {
    predict_p.innerText = "麦克风权限被拒绝或出错";
    console.error("录音错误：", err);
    recordBtn.disabled = false;
  }
});

// normalize tensor 数据
function normalize(tensor, min, max) {
  return tf.tidy(() => {
    const MIN_VALUES = min || tf.min(tensor, 0);
    const MAX_VALUES = max || tf.max(tensor, 0);
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);
    return { NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES };
  });
}
