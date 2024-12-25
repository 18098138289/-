import os
import re
import numpy as np
import pandas as pd

# 传统机器学习
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# 深度学习 / CNN
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                     Dense, Dropout, Input, GlobalAveragePooling2D)
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.applications.resnet import ResNet50, preprocess_input as resnet_preprocess



# LBP (Local Binary Patterns) 特征
# pip install scikit-image
from skimage.feature import local_binary_pattern

###############################################################################
# 全局配置
###############################################################################

# 原始人脸图像规格 (raw 数据是 128×128 灰度)
IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 1

# 训练参数
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

# 数据路径 (请根据实际情况修改)
TRAIN_LABEL_PATH = r"C:\Users\86180\PycharmProjects\pythonProject\人脸图像识别\face\faceDR"
TEST_LABEL_PATH = r"C:\Users\86180\PycharmProjects\pythonProject\人脸图像识别\face\faceDS"
RAWDATA_DIR = r"C:\Users\86180\PycharmProjects\pythonProject\人脸图像识别\face\rawdata"

# 有缺失/错误 face_id
MISSING_FACE_IDS = [1228, 1808, 4056, 4135, 4136, 5004]


###############################################################################
# 1. 数据加载：解析 faceDR/faceDS + 读取 rawdata
###############################################################################

def parse_face_line(line: str):
    """
    从一行文本, 例如:
      1223 (_sex  male) (_age  child) (_race white) ...
    中提取 face_id, gender, age, race, expression 等.
    """
    line = line.strip()
    if not line:
        return None

    match_id = re.match(r'^(\d+)\s+', line)
    if not match_id:
        return None
    face_id = match_id.group(1)

    rest = line[match_id.end():]

    row_result = {
        "face_id": face_id,
        "gender": None
    }
    # 用正则找 _sex
    pattern_gender = re.search(r'\(_sex\s+([^)]+)\)', rest)
    if pattern_gender:
        row_result["gender"] = pattern_gender.group(1).strip()
    return row_result


def load_labels(file_path):
    """逐行解析 faceDR/faceDS 文件, 返回包含 [face_id, gender] 的 DataFrame."""
    parsed_rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = parse_face_line(line)
            if row:
                parsed_rows.append(row)
    df = pd.DataFrame(parsed_rows)

    # 清洗
    df = df.dropna(subset=["face_id", "gender"])
    df["face_id"] = df["face_id"].astype(str)
    df = df[~df["face_id"].isin([str(i) for i in MISSING_FACE_IDS])]
    df["gender"] = df["gender"].replace({"m": "male", "f": "female"})
    return df


def read_raw_face(file_path, height=128, width=128):
    """读取 128×128 8-bit灰度 raw 文件."""
    if not os.path.isfile(file_path):
        return None
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    arr = np.frombuffer(raw_data, dtype=np.uint8)
    if len(arr) != height * width:
        return None
    # 注意 MATLAB reshape(128,128)' 这里需要转置
    img = arr.reshape((height, width)).T
    return img


def load_images_and_labels(df, rawdata_dir):
    """ 根据 df(含 face_id, gender), 从 rawdata_dir 读取图像, 返回 X, y. """
    images = []
    labels = []
    for _, row in df.iterrows():
        face_id = row["face_id"]
        gender = row["gender"]
        file_path = os.path.join(rawdata_dir, face_id)
        img = read_raw_face(file_path, IMG_HEIGHT, IMG_WIDTH)
        if img is None:
            continue

        images.append(img)
        labels.append(gender)
    X = np.array(images, dtype=np.uint8)  # shape (N, 128, 128), uint8
    y = np.array(labels)
    return X, y


###############################################################################
# 2. 方法1：LBP + SVM
###############################################################################
def extract_lbp_features(img, P=8, R=1):
    """
    对单张灰度图计算 LBP 特征 (示例).
    skimage.feature.local_binary_pattern:
      radius=R, n_points=P*R
    返回图像大小相同的LBP特征图.
    为了后续分类, 一般再统计直方图 (见下).
    """
    lbp = local_binary_pattern(img, P * R, R, method='uniform')
    # 统计直方图 (uniform LBP 的 bin 数量一般是 P*R+2)
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, P * R + 3),
                           range=(0, P * R + 2))
    # 归一化
    hist = hist.astype(float) / (hist.sum() + 1e-7)
    return hist  # shape (P*R+2,)


def lbp_svm_pipeline(X_train, y_train, X_test, y_test):
    """
    训练 LBP + SVM, 并在测试集上评估.
    X_train/X_test: 灰度图 (N, 128, 128), dtype=uint8
    """
    # 提取 LBP 特征
    X_train_feats = []
    for img in X_train:
        feat = extract_lbp_features(img, P=8, R=1)
        X_train_feats.append(feat)
    X_train_feats = np.array(X_train_feats)

    X_test_feats = []
    for img in X_test:
        feat = extract_lbp_features(img, P=8, R=1)
        X_test_feats.append(feat)
    X_test_feats = np.array(X_test_feats)

    # 标签编码
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)  # 0/1
    y_test_enc = le.transform(y_test)

    # 训练 SVM (RBF)
    svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=False)
    svm_clf.fit(X_train_feats, y_train_enc)

    # 预测
    y_pred = svm_clf.predict(X_test_feats)
    acc = accuracy_score(y_test_enc, y_pred)
    print("[LBP+SVM] 测试集准确率: %.4f" % acc)
    print("分类报告:")
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
    print("混淆矩阵:")
    print(confusion_matrix(y_test_enc, y_pred))


###############################################################################
# 3. 方法2：PCA + k-NN
###############################################################################
def pca_knn_pipeline(X_train, y_train, X_test, y_test, n_components=100, k=3):
    """
    训练 PCA + k-NN, 并在测试集上评估.
    X_train/X_test: 灰度图 (N, 128, 128)
    """
    # reshape: (N, 128*128)
    n_samples, h, w = X_train.shape
    X_train_reshaped = X_train.reshape(n_samples, -1).astype(np.float32)

    n_samples_test = X_test.shape[0]
    X_test_reshaped = X_test.reshape(n_samples_test, -1).astype(np.float32)

    # 标签编码
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # PCA
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    pca.fit(X_train_reshaped)
    X_train_pca = pca.transform(X_train_reshaped)
    X_test_pca = pca.transform(X_test_reshaped)

    # k-NN
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train_pca, y_train_enc)

    # 预测
    y_pred = knn_clf.predict(X_test_pca)
    acc = accuracy_score(y_test_enc, y_pred)
    print("[PCA+%d-NN] 测试集准确率: %.4f" % (k, acc))
    print("分类报告:")
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
    print("混淆矩阵:")
    print(confusion_matrix(y_test_enc, y_pred))


###############################################################################
# 4. 方法3：自定义小型 CNN
###############################################################################
def build_simple_cnn(input_shape=(128, 128, 1)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))  # 二分类
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def cnn_pipeline(X_train, y_train, X_test, y_test):
    """
    训练一个简单 CNN, 在测试集上评估.
    X_train/X_test: (N, 128, 128), 灰度
    """
    # 把图像扩展维度到 (N, 128,128,1), 并归一化 [0,1]
    X_train_cnn = X_train[..., np.newaxis].astype(np.float32) / 255.0
    X_test_cnn = X_test[..., np.newaxis].astype(np.float32) / 255.0

    # 标签编码 + 独热
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    y_train_onehot = to_categorical(y_train_enc, num_classes=2)
    y_test_onehot = to_categorical(y_test_enc, num_classes=2)

    # 划分验证集
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_cnn, y_train_onehot, test_size=0.2, random_state=2023
    )

    # 构建并训练 CNN
    model = build_simple_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    model.summary()
    model.fit(X_tr, y_tr,
              validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # 评估
    loss_test, acc_test = model.evaluate(X_test_cnn, y_test_onehot, verbose=0)
    print("[Simple CNN] 测试集准确率: %.4f" % acc_test)

    y_pred_prob = model.predict(X_test_cnn)
    y_pred = np.argmax(y_pred_prob, axis=1)
    print("分类报告:")
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
    print("混淆矩阵:")
    print(confusion_matrix(y_test_enc, y_pred))


###############################################################################
# 5. 方法4：预训练模型 (ResNet50)
###############################################################################
def build_resnet50_classifier(num_classes=2):
    """
    使用预训练 ResNet50 做特征提取，然后接一个全连接层做分类.
    Input shape: (224,224,3)
    """
    base_model = ResNet50(weights='imagenet', include_top=False,
                          input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    #  Dense层
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=out)
    # 如果要 Fine-tune, 可设置 base_model.trainable = True
    # 这里演示只 train 顶层
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def resnet_pipeline(X_train, y_train, X_test, y_test):
    """
    用 ResNet50 做预训练, 需把灰度图扩展为 RGB 并 resize 为 (224,224).
    """
    # 1) resize + 扩展到3通道
    import cv2

    def preprocess_resnet(img_array):
        # img_array shape: (128,128)
        # 先 resize (224,224), 再扩成3通道
        img_resized = cv2.resize(img_array, (224, 224),
                                 interpolation=cv2.INTER_AREA)
        img_3ch = np.stack([img_resized] * 3, axis=-1)  # (224,224,3)
        # 变成 float 并预处理
        img_3ch = img_3ch.astype(np.float32)
        return img_3ch

    X_train_resnet = np.array([preprocess_resnet(img) for img in X_train])
    X_test_resnet = np.array([preprocess_resnet(img) for img in X_test])

    # ResNet 图像归一化: resnet_preprocess
    X_train_resnet = resnet_preprocess(X_train_resnet)
    X_test_resnet = resnet_preprocess(X_test_resnet)

    # 标签编码 + 独热
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    y_train_onehot = to_categorical(y_train_enc, num_classes=2)
    y_test_onehot = to_categorical(y_test_enc, num_classes=2)

    # 2) 构建 ResNet50 模型
    model = build_resnet50_classifier(num_classes=2)
    model.summary()

    # 3) 划分验证集
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_resnet, y_train_onehot, test_size=0.2, random_state=2023
    )

    # 4) 训练
    model.fit(X_tr, y_tr,
              validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # 5) 测试评估
    loss_test, acc_test = model.evaluate(X_test_resnet, y_test_onehot, verbose=0)
    print("[ResNet50] 测试集准确率: %.4f" % acc_test)

    y_pred_prob = model.predict(X_test_resnet)
    y_pred = np.argmax(y_pred_prob, axis=1)
    print("分类报告:")
    print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
    print("混淆矩阵:")
    print(confusion_matrix(y_test_enc, y_pred))


###############################################################################
# 6. 主流程
###############################################################################
def main():
    # 读取标签
    df_train = load_labels(TRAIN_LABEL_PATH)
    df_test = load_labels(TEST_LABEL_PATH)
    print("训练集文本记录数:", len(df_train))
    print("测试集文本记录数:", len(df_test))

    # 加载图像
    X_train, y_train = load_images_and_labels(df_train, RAWDATA_DIR)
    X_test, y_test = load_images_and_labels(df_test, RAWDATA_DIR)
    print("训练集图像数量:", X_train.shape, "训练集标签数量:", y_train.shape)
    print("测试集图像数量:", X_test.shape, "测试集标签数量:", y_test.shape)

    # 若数据过少，这里仅做演示
    if len(X_train) == 0:
        print("错误: 训练集为空，无法继续。")
        return

    # 方法1：LBP + SVM
    lbp_svm_pipeline(X_train, y_train, X_test, y_test)
    print("=" * 60)

    # 方法2：PCA + k-NN
    pca_knn_pipeline(X_train, y_train, X_test, y_test,
                     n_components=100, k=3)
    print("=" * 60)

    # 方法3：自定义小型 CNN
    cnn_pipeline(X_train, y_train, X_test, y_test)
    print("=" * 60)

    # 方法4：预训练 ResNet50
    resnet_pipeline(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
