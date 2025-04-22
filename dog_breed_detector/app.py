from flask import Flask, render_template
import eventlet
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np
import torchvision.models as models
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask import jsonify

eventlet.monkey_patch()
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # 允許所有來源
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# 加載模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

breed_model = torch.load(r'C:\Ljf\大學7788\專題\dog\.venv\Lib\resnet50_dog_breed1.pth', map_location=device)
breed_model.eval()

num_classes = 4  # 你的狗狗情緒分類數量
emotion_model = models.resnet50(pretrained=False)
emotion_model.fc = nn.Linear(emotion_model.fc.in_features, num_classes)
emotion_model.load_state_dict(torch.load(r"C:\Ljf\大學7788\專題\dog\.venv\Lib\dog_emotion_resnet50_v1.pth", map_location=device))
emotion_model = emotion_model.to(device)
emotion_model.eval()

# 影像轉換
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

breed_translation = {
    "affenpinscher": "阿芬犬",
    "afghan_hound": "阿富汗獵犬",
    "african_hunting_dog": "非洲獵犬",
    "airedale": "艾爾戴爾梗",
    "american_staffordshire_terrier": "美國斯塔福郡梗",
    "appenzeller": "阿彭策爾犬",
    "australian_terrier": "澳洲梗",
    "basenji": "巴仙吉犬",
    "basset": "巴吉度獵犬",
    "beagle": "米格魯",
    "bedlington_terrier": "貝林頓梗",
    "bernese_mountain_dog": "伯恩山犬",
    "black-and-tan_coonhound": "黑褐浣熊獵犬",
    "blenheim_spaniel": "布倫海姆獵犬",
    "bloodhound": "尋血獵犬",
    "bluetick": "藍色滴答獵犬",
    "border_collie": "邊境牧羊犬",
    "border_terrier": "邊境梗",
    "borzoi": "俄羅斯獵狼犬",
    "boston_bull": "波士頓鬥牛犬",
    "bouvier_des_flandres": "法蘭德斯畜牧犬",
    "boxer": "拳師犬",
    "brabancon_griffon": "布拉班特獅毛犬",
    "briard": "布里亞德犬",
    "brittany_spaniel": "布列塔尼獵犬",
    "bull_mastiff": "牛頭獒",
    "cairn": "凱恩梗",
    "cardigan": "卡迪根柯基",
    "chesapeake_bay_retriever": "切薩皮克灣獵犬",
    "chihuahua": "吉娃娃",
    "chow": "松獅犬",
    "clumber": "克倫伯犬",
    "cocker_spaniel": "可卡獵犬",
    "collie": "柯利牧羊犬",
    "curly-coated_retriever": "捲毛獵犬",
    "dandie_dinmont": "丹迪丁蒙梗",
    "dhole": "印度野犬",
    "dingo": "澳洲野犬",
    "doberman": "杜賓犬",
    "english_foxhound": "英國獵狐犬",
    "english_setter": "英國獵鷹犬",
    "english_springer": "英國獵犬",
    "entlebucher": "恩特布赫犬",
    "eskimo_dog": "愛斯基摩犬",
    "flat-coated_retriever": "平毛獵犬",
    "french_bulldog": "法國鬥牛犬",
    "german_shepherd": "德國牧羊犬",
    "german_short-haired_pointer": "德國短毛指示犬",
    "giant_schnauzer": "巨型雪納瑞犬",
    "golden_retriever": "黃金獵犬",
    "gordon_setter": "戈登獵犬",
    "great_dane": "大丹犬",
    "great_pyrenees": "大比利牛犬",
    "greater_swiss_mountain_dog": "大瑞士山犬",
    "groenendael": "格倫登達爾犬",
    "ibizan_hound": "伊比沙獵犬",
    "irish_setter": "愛爾蘭獵鷹犬",
    "irish_terrier": "愛爾蘭梗",
    "irish_water_spaniel": "愛爾蘭水獵犬",
    "irish_wolfhound": "愛爾蘭獵狼犬",
    "italian_greyhound": "意大利靈緝犬",
    "japanese_spaniel": "日本獵犬",
    "keeshond": "基士犬",
    "kelpie": "凱爾比犬",
    "kerry_blue_terrier": "凱利藍梗",
    "komondor": "科蒙多犬",
    "kuvasz": "庫瓦茲犬",
    "labrador_retriever": "拉布拉多獵犬",
    "lakeland_terrier": "雷克蘭梗",
    "leonberg": "萊昂貝格犬",
    "lhasa": "拉薩犬",
    "malamute": "馬拉穆特犬",
    "malinois": "比利時牧羊犬",
    "maltese_dog": "馬爾濟斯犬",
    "mexican_hairless": "墨西哥無毛犬",
    "miniature_pinscher": "迷你品犬",
    "miniature_poodle": "迷你貴賓犬",
    "miniature_schnauzer": "迷你雪納瑞",
    "newfoundland": "紐芬蘭犬",
    "norfolk_terrier": "諾福克梗",
    "norwegian_elkhound": "挪威獵鹿犬",
    "norwich_terrier": "諾里奇梗",
    "old_english_sheepdog": "古英國牧羊犬",
    "otterhound": "獺獵犬",
    "papillon": "蝴蝶犬",
    "pekinese": "北京犬",
    "pembroke": "彭布羅克犬",
    "pomeranian": "博美犬",
    "pug": "巴哥犬",
    "redbone": "紅骨獵犬",
    "rhodesian_ridgeback": "羅德西亞脊背犬",
    "rottweiler": "羅威納犬",
    "saint_bernard": "聖伯納犬",
    "saluki": "薩路基犬",
    "samoyed": "薩摩耶犬",
    "schipperke": "席普克犬",
    "scotch_terrier": "蘇格蘭梗",
    "scottish_deerhound": "蘇格蘭獵鹿犬",
    "sealyham_terrier": "西里哈姆梗",
    "shetland_sheepdog": "喜樂蒂牧羊犬",
    "shih-tzu": "西施犬",
    "siberian_husky": "西伯利亞哈士奇",
    "silky_terrier": "絲滑梗",
    "soft-coated_wheaten_terrier": "軟毛小麥梗",
    "staffordshire_bullterrier": "斯塔福郡鬥牛犬",
    "standard_poodle": "標準貴賓犬",
    "standard_schnauzer": "標準雪納瑞",
    "sussex_spaniel": "薩塞克斯獵犬",
    "tibetan_mastiff": "西藏獒犬",
    "tibetan_terrier": "西藏梗",
    "toy_poodle": "迷你貴賓犬",
    "toy_terrier": "玩具梗",
    "vizsla": "維茲拉犬",
    "walker_hound": "沃克獵犬",
    "weimaraner": "魏瑪獵犬",
    "welsh_springer_spaniel": "威爾士春獵犬",
    "west_highland_white_terrier": "西高地白梗",
    "whippet": "靈緝犬",
    "wire-haired_fox_terrier": "剛毛狐狸梗",
    "yorkshire_terrier": "約克夏梗"
}

#情绪分类中英文标签
emotion_labels = ["angry", "fearful", "happy", "relaxed"]

emotion_translation = {
    "angry": "生气",
    "fearful": "害怕",
    "happy": "开心",
    "relaxed": "放松",
}

@app.route('/')
def index():
    return render_template('index2.html')

@socketio.on('predict_image')
def handle_predict_image(data):

    try:
        # 將 base64 圖片轉換為 PIL 格式
        print("收到影像資料！")
        image_data = data['image'].split(",")[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        print("影像轉換完成！")
        # 模型預測
        with torch.no_grad():
            breed_output = breed_model(image_tensor)  # 預測狗狗品種
            emotion_output = emotion_model(image_tensor)
            print(emotion_output)
        # 假設這裡有對應的分類邏輯
        breed_pred = torch.argmax(breed_output, dim=1).item()
        emotion_pred = torch.argmax(emotion_output, dim=1).item()

        breed_pred_index = list(breed_translation.values())[breed_pred]  # 转换为中文品种
        emotion_pred_index = emotion_translation[emotion_labels[emotion_pred]]  # 转换为中文情绪
        # 映射結果到實際的品種和情緒名稱
        print(f"預測完成：品種 {breed_pred_index}, 情緒 {emotion_pred_index}")

        # 返回預測結果
        emit('predict_result', {
            'breed': breed_pred_index,
            'emotion': emotion_pred_index
        })
        print("已發送結果到前端！")

    except Exception as e:
        print(f"發生錯誤：{e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port)
