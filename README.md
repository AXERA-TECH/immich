# 一、项目简介

Immich 是一款支持本地 AI 检索与多端备份的开源自托管智能相册。将其深度适配并独立部署在 AX8850 边缘算力板卡上，可让所有照片数据与 AI 推理均在本地物理隔离运行，从而实现100%的数据主权与最高级别的隐私安全保障。

**原项目官方资源：**

- 官方网站：[https://immich.app/](https://immich.app/)
- GitHub 仓库：[https://github.com/immich-app/immich](https://github.com/immich-app/immich)

本项目的目标是演示如何在 AX8850 开发平台上部署 Immich，部署后可直接使用 AX8850 板卡，或者外接 **AI 算力卡** 进行使用。

# 二、部署包

获取特定历史版本包，或查看完整的更新日志，请前往 **[Releases 归档页面](https://github.com/AXERA-TECH/immich/releases)** 查找。

| immich-axera-aarch64-*.tar.gz | 适用于 AX650 、 AX8850 平台 |
| --- | --- |
| immich-axera-x86-*.tar.gz | 适用于群晖X86平台 |
| immich-axera-models-*.tar.gz | 可选离线模型包 |

# 三、支持平台

- **AX650 / AX8850**
    - **AX650N DEMO Board**
    - [**M4N-Dock(爱芯派Pro)**](https://wiki.sipeed.com/hardware/zh/maixIV/m4ndock/m4ndock.html)
    - [**M.2 Accelerator card**](https://axcl-docs.readthedocs.io/zh-cn/latest/doc_introduction.html#id4)
- **群晖+算力卡**

# 四、支持模型

在 Immich 项目中，针对爱芯元智（Axera）AI 加速卡优化后的预编译模型，可参考以下版本获取：

- [AXERA-TECH/ViT-L-14-336-CN__axera](https://huggingface.co/AXERA-TECH/ViT-L-14-336-CN__axera)
- [AXERA-TECH/PPOCR_v5](https://huggingface.co/AXERA-TECH/PPOCR_v5/tree/main)
- [AXERA-TECH/Insightface](https://huggingface.co/AXERA-TECH/Insightface/tree/main)

# 五、部署操作

目前提供 AArch64 和 X86 两个版本的部署包，使用流程一致，操作时只需要注意区分包名即可。

推荐群晖使用 Docker 安装 Server 和 ML 服务。

AX8850 板卡等选择 AArch64 ，Docker 安装 Server ，其中 ML 服务安装方式可三选一，推荐 Docker安装更快捷。

## 1. 系统配置与代码获取

### （1）扩充存储空间（可选）

- **挂载 SD Card**：若系统内部空间不够，可选择挂载 SD Card 来存放数据。首先需将 SD Card 格式化为 `ext4` 格式并挂载。

```bash
lsblk

#格式化SD Card  注意保存数据
# mkfs.ext4 /dev/mmcblk1p1

mkdir /mnt/sdcard
mount /dev/mmcblk1p1 /mnt/sdcard
```

- **配置开机自动挂载**：通过 `blkid /dev/mmcblk1p1` 获取 UUID 后，修改 `/etc/fstab` 文件以实现开机自启。配置示例如 `UUID=13054439-9d76-47ca-b8ed-c7cd8d208a5d /mnt/sdcard ext4 defaults 0 2`。

### （2） 获取部署代码（必做）

- 下载部署包到平台，放到本地 Immich 文件，目前提供 X86 和 AArch64 版本，以 v3.0.1 为例。

```bash
mkdir immich
cd immich

#拉取部署包（从第二章节链接获取）immich-axera-aarch64-v3.0.1.tar.gz 或 immich-axera-x86-v3.0.1.tar.gz到本地并解压
#AArch64版本
tar -xzvf immich-axera-aarch64-v3.0.1.tar.gz
#X86版本
tar -xzvf immich-axera-x86-v3.0.1.tar.gz 

```

解压后结构类似：

```bash
immich
|-- deploy
|   |-- docker-compose.axcl.yml
|   |-- docker-compose.ml.yml
|   |-- docker-compose.yml
|   |-- .env
|   |-- library
|   `-- postgres
|-- huggingface
|-- images
|   |-- ax-immich-ml-aarch64.tar.gz
|   |-- ax-immich-server-aarch64.tar.gz
|   |-- immich-postgres-aarch64.tar.gz
|   `-- valkey-8-bookworm-aarch64.tar.gz
|-- models
`-- packages
    |-- axengine-0.1.3-py3-none-any.whl
    |-- immich_ml-3.0.1-py3-none-any.whl
    `-- requirements.txt
```

## 2.启动server服务

### （1）检查env文件（必做）

- 发布包默认包含 `deploy/.env`。首次启动前请检查路径、数据库密码等配置是否符合当前设备。

```bash
vi deploy/.env
```

### （2）Docker 环境配置（可选）

- **修改数据目录**：若内部空间不够，可修改 Docker 镜像路径至 SD Card。创建 `/mnt/sdcard/docker_data` 目录后，修改 `/lib/systemd/system/docker.service`，在 `ExecStart` 后面添加 `--data-root=/mnt/sdcard/docker_data` 并重启服务。请务必确保 SD Card 已配置开机自动挂载。

### （3）启动 Server 服务（必做）

- 加载 Docker 镜像包：`images` 目录包含 Server、Postgres、Redis 以及 ML 镜像。启动 Server 服务前，需先导入 Server、Postgres、Redis 镜像。

```bash
#加载server所需镜像
docker load -i images/ax-immich-server-x86.tar.gz
docker load -i images/immich-postgres-x86.tar.gz
docker load -i images/valkey-8-bookworm-x86.tar.gz
```

- 并使用 Docker Compose 启动容器服务,会启动`immich_server`、 `immich_postgres`、 `immich_redis` 。

```bash
docker compose -f deploy/docker-compose.yml up -d
```

- 可通过 `docker ps` 查看容器状态。若长时间处于 `starting` 状态，建议使用 `docker logs immich_server --tail 20` 等命令排查启动日志。
- 启动正常后，通过浏览器访问 `http://<开发板IP>:2283` 即可进入网页端进行注册和登录。

![image.png](./design/Axera/immich.png)

## 3. 机器学习 (ML) 服务部署

Server 启动后可访问网页，但是无法使用智能搜图、OCR、人脸识别等功能，需要启动ml服务。

这里推荐直接使用Docker服务快速部署，但也提供直接安装和虚拟环境安装方案以便满足不同需求。**以下方案三选一即可（NAS只支持 Docker 方案并注意核对axcl关键接口映射）：**

### （1）方案一：使用Docker

- **获取并安装模型**：Docker 建议默认使用本地模型

```bash
#下载模型包immich-axera-models-v3.0.1,从第二章节链接获取

#将模型放到models目录解压
tar -xzvf immich-axera-models-v3.0.1.tar.gz

#解压后三个文件：clip、facial-recognition、ocr

models/
|-- clip
|   `-- ViT-L-14-336-CN__axera
|-- facial-recognition
|   `-- buffalo_l__axera
`-- ocr
    `-- PPOCR_v5__axera
```

- **加载并启动服务**

```bash
#加载镜像
#AArch64
docker load -i images/ax-immich-ml-aarch64.tar.gz
#X86
docker load -i images/ax-immich-ml-x86.tar.gz

#启动镜像（这里通过docker-compose.axcl 进行axcl端口映射，如不使用axcl可以不加载docker-compose.axcl.yml）
docker compose -f deploy/docker-compose.ml.yml -f deploy/docker-compose.axcl.yml up -d

#比如在AX8850板卡使用AxEngineExecutionProvider，则不用使用deploy/docker-compose.axcl.yml
docker compose -f deploy/docker-compose.ml.yml up -d
```

- **算力卡检测**：若需要使用算力卡，通过指令 axcl-smi 确定算力卡是否正常，可见类似输出:、

```bash
root@ax650:/opt/bin/axcl# ./axcl-smi
+------------------------------------------------------------------------------------------------+
| AXCL-SMI  V3.15.0_20260609020154                                Driver  V3.15.0_20260609020154 |
+-----------------------------------------+--------------+---------------------------------------+
| Card  Name                     Firmware | Bus-Id       |                          Memory-Usage |
| Fan   Temp                Pwr:Usage/Cap | CPU      NPU |                             CMM-Usage |
|=========================================+==============+=======================================|
|    0  AX8850                    V3.15.0 | 0001:81:00.0 |                148 MiB /      945 MiB |
|   --   39C                      -- / -- | 1%        0% |                 18 MiB /     7040 MiB |
+-----------------------------------------+--------------+---------------------------------------+

+------------------------------------------------------------------------------------------------+
| Processes:                                                                                     |
| Card      PID  Process Name                                                   NPU Memory Usage |
|================================================================================================|
```

### （2）方案二：直接安装ml服务

- 安装所需依赖和Wheel

```bash
#注意安装py 3.11版本(immich官方指定)
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# 安装依赖
pip install -r packages/requirements.txt

#建议选择PyAXEngine最新版本 当前最新0.1.3.rc3
pip install packages/axengine-0.1.3-py3-none-any.whl

#安装Wheel包
pip install packages/immich_ml-3.0.1-py3-none-any.whl
```

- 启动ml服务：默认启动不使用本地模型

```bash
IMMICH_HOST=0.0.0.0 IMMICH_PORT=3003 python -m immich_ml
```

可选参数：

`MACHINE_LEARNING_CACHE_FOLDER` ：指定模型缓存位置。模型默认位置在 models 文件中

`HF_HOME` ：指定hf位置，默认位置在 huggingface 文件中

`HF_HUB_OFFLINE` ：指定启动方式，1为本地启动。Docker默认本地启动，默认不下载模型。其他启动方式默认值为0，当初次使用模型时，会下载远程模型，可以通过 `HF_HUB_OFFLINE=1` 加上模型包 immich-axera-models.zip 实现本地模型启动。

### （3）方案三：虚拟环境安装

- 若主存储空间不够，可安装 `python3.11-venv` 并在 SD Card 上创建虚拟环境 `/mnt/sdcard/immich_venv` 以节省空间。

```bash
apt-get install -y python3.11-venv
python3.11 -m venv /mnt/sdcard/immich_venv
```

- 了避免存储不够，安装时可通过 `TMPDIR` 和 `-cache-dir` 参数将临时文件与缓存放到 SD Card 上。

```bash
# 激活虚拟环境（若使用了venv）
source /mnt/sdcard/immich_venv/bin/activate

# 安装依赖
TMPDIR=/mnt/sdcard pip install -r packages/requirements.txt --cache-dir /mnt/sdcard/.pip_cache

#建议选择PyAXEngine最新版本 当前最新0.1.3.rc3
TMPDIR=/mnt/sdcard pip install packages/axengine-0.1.3-py3-none-any.whl --cache-dir /mnt/sdcard/.pip_cache

#安装Wheel包
TMPDIR=/mnt/sdcard pip install packages/immich_ml-3.0.1-py3-none-any.whl --cache-dir /mnt/sdcard/.pip_cache
```

- 启动方式和可选参数与 直接安装 ml 服务相同

```bash
#例如指定缓存路径，使用本地模型启动
mkdir -p /mnt/sdcard/immich_ml_cache /mnt/sdcard/huggingface
MACHINE_LEARNING_CACHE_FOLDER=/mnt/sdcard/immich_ml_cache HF_HUB_OFFLINE=1 HF_HOME=/mnt/sdcard/huggingface IMMICH_HOST=0.0.0.0 IMMICH_PORT=3003 python -m immich_ml

```

## 4. 功能验证与设置

APP和网页端功能类似，这里只做部分关键功能说明，更多功能可自行探索。

### （1）查询网关 IP：

在宿主机通过命令 查看 IP（通常类似于 `172.19.0.1`）。

```bash
 #AArch64
 docker network inspect immich-axera-aarch64_default | grep Gate
 #X86
 docker network inspect immich-axera-x86_default | grep Gate
```

### （2）配置 ML 服务：

登录 Immich **网页端**，点击左上角**头像**，选择 **Administration**，进入**设置**，选择**机器学习设置**，配置上一步获取到的 IP，模型默认为 Axera 适配模型，关键参数配置根据自行需求进行填写。

CLIP模型：  [ViT-L-14-336-CN__axera](https://huggingface.co/AXERA-TECH/ViT-L-14-336-CN__axera)

人脸识别模型：[buffalo_l_axera(AXERA)](https://huggingface.co/AXERA-TECH/Insightface)

OCR模型：[PPOCR_v5_axera(AXERA)](https://huggingface.co/AXERA-TECH/PPOCR_v5/blob/main/README-zh.md)

![image.png](./design/Axera/4.2.1.png)

### （3）智能搜索功能：

- 在网页端上传图片
- 创建智能搜索任务
- 任务处理完毕后，执行智能搜索以验证 ML 服务是否正常解析图像特征。

![image.png](./design/Axera/4.3.1.png)

![image.png](./design/Axera/4.3.2.png)

### （4）APP智能搜索

- 下载App，并使手机在同一网络环境下
- 填写Ip和账号密码进入
- 进行智能搜索

![image.png](./design/Axera/4.4.1.jpg)

### （5）人脸识别自动构建相册

- 上传人物图片，跑人脸检测和人脸识别任务
- 当同一人物图片数量超过模型配置时配置的“最小识别数量”即可自动生成人物相册
- 可在探索界面，给人物进行命名
- 后续新添加已知人物的照片，将自动识别后放入对应的人物相册

![image.png](./design/Axera/4.5.1.png)


### （6）OCR识别

- 上传人物图片，跑OCR任务
- 通过OCR功能搜索图片

![image.png](./design/Axera/4.6.1.png)


### （7）新图片加入

- 当首次跑完OCR、人脸识别、智能搜索等任务后，若增加新的图片，会自动对图片进行数据处理，无需二次启动任务
- 若出现相同图像，会自动进行重复项处理，避免图片重复
- 若传入的是人脸图片，会自动生成人物相册，且根据地址信息放入地图，如传入了3张新的梅西照片，其中一张带有地址信息。

![image.png](./design/Axera/4.7.1.png)

![image.png](./design/Axera/4.7.2.png)


### （8）APP自动同步相册

- 当APP配置了相册同步功能后，当检测到相册有新图片加入，会自动进行数据同步
- APP数据同步需要在同一网络下
- 可直接关联相机相册，拍照后即可上传备份，避免数据丢失

![app备份.jpg](./design/Axera/4.8.1.jpg)

# 六、性能评估

Immich 服务由 Docker 中的 `server/postgres/redis` 与宿主机 `immich_ml` 组成。由于模型懒加载、运行时缓存、数据库缓存以及 AX CMM 分配策略等因素，MEM 与 CMM 占用会受操作影响。下面固定测试场景进行性能评估：

本次测试模拟用户日常使用流程,使用 AxEngineExecutionProvider，使用虚拟环境部署：通过 Web 页面批量上传图片，随后由系统自动完成缩略图、元数据、人脸识别、OCR、重复检测和智能搜索索引等后台处理。

- 测试图片：2310 张
- 上传方式：Web 页面手动上传
- ML 模型：本地缓存加载，`HF_HUB_OFFLINE=1`
- 测试模式：预热后的日常连续使用状态

## 1.整体处理能力

上传2310张混合图片

| **阶段** | **耗时** | **处理能力** |
| --- | --- | --- |
| Web 批量上传 | 205 秒 | 约 11.3 张/秒 |
| 上传后后台 AI 处理 | 829 秒 | 约 2.8 张/秒 |
| 从开始上传到全部完成 | 1034 秒 | 约 17 分 14 秒 |

处理完成后，本批图片产生：

- 人脸记录：1660 条
- OCR 文本记录：710 条

## 2.存储占用

| **目录** | **占用** |
| --- | --- |
| AXERA / Immich ML 模型缓存 | 637 MiB |
| HuggingFace 缓存 | 544 KiB |
| Immich 图库目录 | 585 MiB |

本次测试使用本地模型缓存运行，不依赖外网下载，适合边缘设备、内网部署和离线演示环境。

## 3.搜索能力

- **后端搜索接口能力**
使用脚本直接请求 Immich 搜索接口，每个关键词连续执行 10 轮，第 1 轮作为首次查询参考，第 2-10 轮作为热查询统计。该指标反映服务端从接收搜索请求到返回 JSON 结果的耗时，不包含浏览器输入防抖、页面渲染、缩略图加载等前端开销。

| 能力项 | 测试内容 | 测试结果 | 产品化描述 |
| --- | --- | --- | --- |
| **中文语义搜索** | 查询“穿红色衣服的小孩” | 首次约 105 ms，热查询平均约 60 ms | 支持中文自然语言描述检索，可根据人物、颜色、衣着等复合语义查找图片。 |
| **英文语义搜索** | 查询“red clothes child” | 首次约 83 ms，热查询平均约 59 ms | 支持英文自然语言检索，适合中英文混合展示和国际化场景。 |
| **人物类搜索** | 查询“小孩 / child / 孩子” | 热查询平均约 60-63 ms | 可根据人物类别进行语义检索，适合快速查找人物相关照片。 |
| **颜色与衣着搜索** | 查询“红色衣服 / red shirt” | 热查询平均约 60-61 ms | 可识别图片中的颜色和衣着特征，支持更细粒度的图片查找。 |
| **组合条件搜索** | 查询“person in red” | 热查询平均约 59 ms | 支持人物与颜色等多条件组合描述，提升图库检索的直观性。 |
| **查询响应体验** | 每个关键词连续查询 10 轮 | 首次查询约 76-105 ms，热查询约 58-63 ms | 在 2310 张图片规模下，智能搜索响应保持在百毫秒级以内，适合现场演示和日常使用。 |
| **查询稳定性** | 中英文关键词多轮请求 | HTTP 状态均为 200，异常数 0 | 查询接口响应稳定，连续检索过程中未出现请求失败。 |

## 4.推荐部署余量

基于本轮 2310 张图片测试结果，建议产品化部署预留以下资源：

| **项目** | **建议值** | **说明** |
| --- | --- | --- |
| OS 内存 | >= 3.5 GiB | 按实测总内存峰值约 2.62 GiB 增加 30% 余量 |
| CMM | >= 1.0 GiB | 按实测 CMM 峰值约 732 MiB 增加 30% 余量 |
| 推荐 CMM 配置 | 4 GiB | 本次测试配置余量充足，适合展示和扩展 |
| 模型缓存空间 | >= 1 GiB | 当前模型缓存约 637 MiB，建议留足更新空间 |
| 图库空间 | 原图空间 + 30% 以上 | 缩略图、索引、派生文件会随图库增长 |
