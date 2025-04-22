const socket = io('http://localhost:5000');  // 連接後端 SocketIO

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const result = document.getElementById('result');
const captureButton = document.getElementById('uploadBtn');  // 按鈕

// 啟動攝影機
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
  })
  .catch(err => {
    console.error("無法啟動攝影機:", err);
    result.innerText = "無法存取攝影機：" + err.message;
  });

// 擷取影像並傳送到後端
captureButton.addEventListener('click', () => {
  const context = canvas.getContext('2d');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  context.drawImage(video, 0, 0);
  const imageBase64 = canvas.toDataURL('image/jpeg');

  result.innerText = "🔍 分析中...";

  socket.emit('predict_image', { image: imageBase64 });
});

// 接收辨識結果
socket.on('predict_result', data => {
  if (data.error) {
    result.innerText = "❌ 錯誤：" + data.error;
  } else {
    result.innerHTML = `🐶 品種：${data.breed}<br>😊 情緒：${data.emotion}`;
  }
});

// 顯示連線狀態
socket.on('connect', () => {
  console.log("✅ 已連接到伺服器");
});

socket.on('disconnect', () => {
  console.log("❌ 與伺服器斷線");
});
