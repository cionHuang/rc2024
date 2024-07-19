# RC2024

## 介绍

本仓库为 SPR RC 分部为 RC2024 赛季创建的视觉组代码托管仓库。

内部开发阶段使用[Gitee](https://gitee.com/HHHSSSHHH/rc2024)进行版本控制，现已同步推送至[Github](https://github.com/cionHuang/rc2024)，查询历史版本请前往Gitee(非最新版本且已停止维护)

本项目使用的硬件包括：
- 两台奥比中光 Astra Pro
- 两台迈德威视工业摄像头
- 一个 USB 转 RS485 工具

**本项目基于 yolov5 模型，通过 OpenVino 调用核显进行加速推理。决策中，使用Q-learning强化学习进行推演。**

### 项目环境

- **Python 版本**: 3.9.x

- **Ubuntu 版本**: 22.04

### 核心文件

- `/newmain.py`: 主程序入口
- `/Utils/tools.py`: 所有用到的库、类、工具函数
- `/Utils/new_vino/new_vinodetect.py`: 基于 OpenVino 加速的 yolov5 推理网络
- `/Utils/policy`: 基于强化学习 Q-learning 的筒仓决策

## 项目结构

```plaintext
├── Camera                      # 奥比中光相机配置文件
├── docs                        # 文档文件
├── Mindvision                  # 迈德威视相机配置文件
├── Models                      # yolo 模型文件
├── OpenNI2                     # 深度摄像头库文件
├── R1                          # 为 R1 开发测试程序
├── Utils                       # 所有使用到的库、类、工具函数
│   ├── tools.py                # 工具函数
│   ├── new_vino
│   │   └── new_vinodetect.py   # 基于 OpenVino 加速的 yolov5 推理网络
│   └── policy                  # 基于强化学习 Q-learning 的筒仓决策
|       ├── main.py             # 决策入口
|       ├── generate_states.py  # 生成所有状态        
|       ├── test.py             # 模拟对战测试
|       └── train.py            # 策略模型训练
├── newmain.py                  # 主程序入口
├── requirements.txt            # 环境配置
├── test.py                     # 测试文件
└── testmain.py                 # 测试文件
```

## 使用说明与注意

1.由于本项目涉及到摄像头、串口等硬件，所以必须连接到所有的硬件才能正确运行

2.除requirements.txt的配置文件外，还需要[安装OpenVino工具库](https://github.com/openvinotoolkit/openvino)

3.[奥比中光驱动与OpenNI2库](https://vcp.developer.orbbec.com.cn/resourceCenter)

4.[迈德威视驱动](https://www.mindvision.com.cn/category/software/)

5.由于策略模型中有超过100M的文件，所以采用Git LFS进行存储，需要安装Git LFS才能正常拉取，[安装方法](https://git-lfs.github.com/)

## 开源许可

本项目采用MIT许可证。详细信息请参阅LICENSE文件

## 鸣谢

### 主要贡献者  
  
- **黄绅豪/CionHuang**：感谢Cion在[核心功能开发与团队协调管理]方面所付出的努力
- **刘浩博/HarborLiu**：感谢Harbor在[基于强化学习Q-learning的筒仓决策]中的辛勤工作

### 贡献者列表

此外，还要感谢RC2024赛季加入的新队员，感谢他们在程序调试、模型训练等过程中的贡献：

- **吴兆芸**
- **范嘉恒**

### 特别鸣谢

- **周骏琦/JimmyZhou**：感谢名誉领队周骏琦/JimmyZhou在本赛季中的极限调车，周师傅永远的神，毕业快乐！（另附[电控代码](https://gitee.com/jimmyzhou226/rc-control-2024)）
- **范文栋**：感谢R2主设范文栋设计的瓦力，同时也是国服第一R1使用者。
- **刘世强、夏东华、陈溪铭**：感谢R1主设刘世强、夏东华、陈溪铭设计的T0R1，真无敌吧！
- **张杰宇**：感谢“指导老师兼摄影”张杰宇，没你朋友圈都没素材，毕业快乐！
- **易昊男**：感谢“唯一指定旗手兼团队心理指导”易昊男，下次别这么早走了，毕业快乐！
- **潘伟骁**：感谢“强大的后援”潘伟骁，古希腊掌管设备的神！毕业快乐！
- **SPR_RC24全体成员**：九楼灯火长明，赛场我们必赢！

## 作者

若有相关问题或交流需求，欢迎邮件联系作者！

- **Cion Huang**: cionhuang124@gmail.com

- **Harbor Liu**: harbourrr123@gmail.com

