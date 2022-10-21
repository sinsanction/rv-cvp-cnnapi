# RV-CVP CNNAPI

RV-CVP 指令集软件编程框架（CNNAPI 编程库）

## 内容

该库分为两个版本 v1 和 v2。cnnapi 目录下为 v1 独有内容，cnnapi_v2 目录下为 v2 独有内容，common 目录下为两者通用内容。一个版本的独有内容加上通用内容为一套完整的 CNNAPI。两个版本可以独立使用，也可以同时使用。

v1 版本支持最低网络每层具有相同精度。v2 版本支持最低网络一层中每列具有相同精度。

## 使用

1. 在 AM （NutShell 的裸机运行环境）中使用

直接使用这里提供的文件，根据 cnnapi_*.h 头文件中的数据结构和 API 构建网络。

这里是一个结合 microbench 测试框架使用 CNNAPI 的用例：[https://github.com/sinsanction/nexus-am/tree/master/apps/cnnapibench](https://github.com/sinsanction/nexus-am/tree/master/apps/cnnapibench)。

2. 在真实环境（比如操作系统提供的标准 C 语言库环境）中使用

删除 cnnapi_*.h 头文件中的 __nutshell_am 宏定义，其他同上。
