#include "Interface/imbaprocess.h"
#include "ui_imbaprocess.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QSettings>
#include <QDebug>
#include <QStandardPaths>
#include <memory>
#include <QString>
#include <QDateTime>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <QTime>
#pragma execution_character_set("utf-8")

//基本全局图像处理变量
namespace BasicImageGlobel {
    bool shouldStop;//数据采集标志位
    QString localFilePath;//图像文件的路径
    cv::Mat histData;//直方图数据
    cv::Mat histimage;//直方图对应的原图
    bool histogramSucc;//直方图计算并绘制成功标志位
    cv::Mat GlobelequalizeHistimge;//全局阈值化原图
    cv::Mat GlobelequalizeHistData;//全局阈值化直方图
    cv::Mat PartequalizeHistimge;//局部阈值化原图
    cv::Mat PartequalizeHistData;//局部阈值化直方图
    cv::Mat ConvolutionaAmbiguityResult;//卷积模糊结果
    cv::Mat CustomFiltering;//自定义滤波结果
    cv::Mat gradientResult;//梯度提取结果
    cv::Mat CannyResult ;//边缘发现结果
    cv::Mat Noiseprocess ;//噪声处理结果
    cv::Mat BliateraResult;//高斯双边处理结果
    cv::Mat SharpenResult;//锐化增强处理
    //二值计算
    cv::Mat ThresholdResult;//图像阈值化分割结果
    cv::Mat GlobalThresholdResult;//全阈值化分割结果
    cv::Mat adaptiveResult;//自适应阈值计算结果
    cv::Mat BinaryResult;//去噪与二值化结果
    //二值分析
    cv::Mat connectComResult;//联通组件计算结果
    cv::Mat FindDrwContours;//发现并绘制轮廓结果
    std::vector<std::vector<cv::Point>>ContoursResult;//轮廓发现的轮廓点集集合
    cv::Mat FittingApproximationResults;//拟合与逼近的结果
    cv::Mat ContoursanalyseResult      ;//轮廓分析结果
    cv::Mat LineDetectionResult        ;//霍夫直线检测结果
    cv::Mat CircleDetectionResult      ;//霍夫圆检测结果
    cv::Mat MinMaxCircleResult         ;//最小外接圆和最大内接圆结果
    cv::Mat ContoursMatchShapResult    ;//轮廓匹配结果
    cv::Mat MaxContourKeyPointCodResult;//最大轮廓与关键点编码结果
    cv::Mat HullDectionResult          ;//凸包检测结果
    //形态学操作
    cv::Mat DilateErodeResult          ;//形态学膨胀和腐蚀操作结果
    cv::Mat OpenCloseOperationResult   ;//形态学开操作和闭操作结果
    cv::Mat GridentResult              ;//形态学梯度结果
    cv::Mat BlackTopHeatResult         ;//顶帽和黑帽结果
    cv::Mat BitMissResult              ;//击中与击不中结果
    cv::Mat CustomDectionResult        ;//自定义结构元素结果
    cv::Mat DictanceChangeResult       ;//距离变换
    //特征提取
    std::vector<cv::Mat>ImagePyramidResult ;//图像金字塔结果
    cv::Mat HarrisCornerResult         ;//Harris角点检测结果
    cv::Mat shiTomasCornerResult       ;//shiTmas角点检测结果
    cv::Mat HOGfeatureDescripResult    ;//HOG描述子检测结果
    cv::Mat ORBMatchingResult          ;//ORB匹配结果
    cv::Mat ObjectDectionResult        ;//特征对象检测结果

}

//将Mat对象转换为Qpixmap对象函数
QPixmap ImbaProcess::matToQPixmap(const cv::Mat &mat)
{
    QPixmap qPixmap;
    if (mat.channels()==3){//如果图像的通道是三通道，那么执行三通道彩色图像转换代码
        //将BGR转换为RGB
        cv::Mat matmid;
        cv::cvtColor(mat, matmid, cv::COLOR_BGR2RGB);
        //将Mat对象转换为 Qimage
        QImage qImage(matmid.data, matmid.cols, matmid.rows, static_cast<int>(matmid.step), QImage::Format_RGB888);
        //将Qimage转换为 qpixmap 并返回
        qPixmap = QPixmap::fromImage(qImage);
    }
    else if(mat.channels()==1){//如果图像是单单通道灰度图像那么执行下面的转换代码
        QImage img(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8);
        qPixmap = QPixmap::fromImage(img);
    }
  return qPixmap;
}

//将Qpixmap对象转换为Mat对象函数
cv::Mat ImbaProcess::QPixmapToMat(const QPixmap &pixmap)
{
    // 将QPixmap转换为QImage
    QImage image = pixmap.toImage();
    // 确保QImage的格式是符合OpenCV需求的
    if (image.format() != QImage::Format_RGB888){
        image = image.convertToFormat(QImage::Format_RGB888);
    }
    // 获取QImage的宽度和高度
    int width = image.width();
    int height = image.height();
    // 将QImage的数据转换为cv::Mat
    cv::Mat mat(height, width, CV_8UC3, const_cast<uchar*>(image.bits()), image.bytesPerLine());
    //将RGB转换为BGR
    cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
    // 由于QImage的数据是按行存储的，所以需要进行深拷贝
    mat = mat.clone();
    return mat;
}

//计算并绘制图像直方图函数
QPixmap calcHistogram(cv::Mat&image){
    cv::Mat Histimage;//直方图画布
    if(image.channels()==3){//输入图像为三通道的
        std::vector<cv::Mat>Graybgr;
        //默认分离BGR通道 Blue:Graybgr[0],Green:Graybgr[1],Red:Graybgr[2]
        split(image,Graybgr);
        //定义直方图处理通道:处理所有通道
        int channel[]={0};
        //创建直方图Mat对象
        cv::Mat r_hist,g_hist,b_hist;
        //bins分辨率
        int histsize[]={255};
        //均匀分布或者不均匀分布的自定义范围
        float r_range[]={0,256};
        const float* ranges[]={r_range};
        //使用计算直方图函数(一维直方图)
        cv::calcHist(&Graybgr[0],1,channel,cv::Mat(),b_hist,1,histsize,ranges,true,false);
        cv::calcHist(&Graybgr[1],1,channel,cv::Mat(),g_hist,1,histsize,ranges,true,false);
        cv::calcHist(&Graybgr[2],1,channel,cv::Mat(),r_hist,1,histsize,ranges,true,false);
        //创建绘制直方图画布
        Histimage=cv::Mat::zeros(600,300,image.type());
        int hist_h=Histimage.rows-10*2;
        //将直方图计算的结果归一化
        cv::normalize(b_hist,b_hist,0,hist_h,cv::NORM_MINMAX,-1,cv::Mat());
        cv::normalize(g_hist,g_hist,0,hist_h,cv::NORM_MINMAX,-1,cv::Mat());
        cv::normalize(r_hist,r_hist,0,hist_h,cv::NORM_MINMAX,-1,cv::Mat());
        //绘制图像直方图
        for(int i=0;i<255;i++){
            cv::line(Histimage,cv::Point(i+10,hist_h-cvRound(b_hist.at<float>(cv::abs(i-1)))),cv::Point(i+1+10,hist_h-cvRound(b_hist.at<float>(i))),cv::Scalar(255,0,0),1,cv::LINE_AA,0);//蓝色
            cv::line(Histimage,cv::Point(i+10,hist_h-cvRound(g_hist.at<float>(cv::abs(i-1)))),cv::Point(i+1+10,hist_h-cvRound(g_hist.at<float>(i))),cv::Scalar(0,255,0),1,cv::LINE_AA,0);//绿色
            cv::line(Histimage,cv::Point(i+10,hist_h-cvRound(r_hist.at<float>(cv::abs(i-1)))),cv::Point(i+1+10,hist_h-cvRound(r_hist.at<float>(i))),cv::Scalar(0,0,255),1,cv::LINE_AA,0);//红色
          }
    }else if(image.channels()==1){//输入图像为单通道的
        //定义直方图处理通道:处理所有通道
        int channel[]={0};
        //创建直方图Mat对象
        cv::Mat gray_hist;
        //bins分辨率
        int histsize[]={255};
        //均匀分布或者不均匀分布的自定义范围
        float r_range[]={0,256};
        const float* ranges[]={r_range};
        //使用计算直方图函数(一维直方图)
        cv::calcHist(&image,1,channel,cv::Mat(),gray_hist,1,histsize,ranges,true,false);
        //创建绘制直方图画布
        Histimage=cv::Mat::zeros(600,300,image.type());
        int hist_h=Histimage.rows-10*2;
        //将直方图计算的结果归一化
        cv::normalize(gray_hist,gray_hist,0,hist_h,cv::NORM_MINMAX,-1,cv::Mat());
        //绘制图像直方图
        for(int i=0;i< 255;i++){
           cv::line(Histimage,cv::Point(i+10,hist_h-cvRound(gray_hist.at<float>(cv::abs(i-1)))),cv::Point(i+1+10,hist_h-cvRound(gray_hist.at<float>(i))),cv::Scalar(255,0,0),1,cv::LINE_AA,0);//蓝色
          }
    }
    std::unique_ptr<ImbaProcess> calcHistogramFunction=std::make_unique<ImbaProcess>();
    return calcHistogramFunction->matToQPixmap(Histimage);
}


//类中采用了静态的（static）的qwidge域或其子类，因为静态和全局对象进入main函数之前就产生了，所以早于main函数里的qapplication对象,所以这里直接给空，需要的时候再实例化使用
std::unique_ptr<ImbaProcess> ImbaProcess::basicImprocess = nullptr;  // 直接实例化静态成员变量

ImbaProcess::ImbaProcess(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ImbaProcess)
{
    ui->setupUi(this);
    //设计界面Logo
    this->setWindowIcon(QIcon(":/new/prefix1/Sourceimage/ImageProcess.png"));
    ui->Lab_Image->clear();
    ui->textEdit_Infor->clear();
    ui->textEdit_Infor->setReadOnly(true);//信息区设置为只读
    ui->textEdit_Infor->setTextColor(QColor(0, 120, 0));//设置文本颜色
    ui->textEdit_Infor->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);//设置垂直滚动条策略
    ui->textEdit_Infor->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);//设置水平滚动条策略
    ui->label_Camera->clear();//视频显示区清除数据
    ui->checkBox_calculate->setCheckState(Qt::Checked);//直方图计算与绘制功能设置默认选中
    ui->doubleSpinBox_contrast->setMaximum(40.0);//局部自适应均衡化最大值
    ui->doubleSpinBox_contrast->setMinimum(0.0);//局部自适应均衡化最小值
    ui->doubleSpinBox_contrast->setValue(4.0);//局部自适应均衡化默认对比度
    ui->spinBox_HSVchannel->setMaximum(2);//HSV均衡化通道最大值
    ui->spinBox_HSVchannel->setMinimum(0);//HSV均衡化通道的最小值
    ui->spinBox_HSVchannel->setValue(2);//设置默认的HSV局部均衡化通道的值，默认是均衡化V（亮度通道）
    this->setWindowTitle(QString("基本数字图像处理"));
    ui->spinBox_ConvolutionaBorder->setMaximum(5);//卷积边界处理方式最大值
    ui->spinBox_ConvolutionaBorder->setMinimum(0);//卷积边界处理方式最小值
    ui->spinBox_ConvolutionaBorder->setValue(4);  //卷积边界处理默认方式(101)
    ui->spinBox_Convolutionamode->setMaximum(2);//卷积方式最大值（均值，高斯，中值）
    ui->spinBox_Convolutionamode->setMinimum(0);//卷积方式最小值
    ui->spinBox_Convolutionamode->setValue(1);  //卷积方式默认值
    ui->spinBox_ScoreSize->setMaximum(99);//卷积核最大值
    ui->spinBox_ScoreSize->setMinimum(3);//卷积核最小值
    ui->spinBox_ScoreSize->setValue(7);//卷积核默认值
    ui->spinBox_specificValue->setValue(25);//自定义滤波水平或者垂直卷积核比值
    ui->spinBox_VH->setMaximum(1);//自定义垂直滤波最大值
    ui->spinBox_VH->setMinimum(0);//自定义垂直滤波最小值
    ui->spinBox_SobelScharr->setMaximum(1);//图像梯度模式选择最大值
    ui->spinBox_SobelScharr->setMinimum(0);//图像梯度模式选择最小值
    ui->spinBox_SobelCoreSize->setValue(3);//Sobel梯度提取卷积核大小
    ui->spinBox_gradientXY->setMaximum(1);//梯度提取方向最大值
    ui->spinBox_gradientXY->setMinimum(0);//梯度提取方向最小值
    ui->spinBox_highThreshold->setMaximum(1000);//边缘发现高阈值最大值
    ui->spinBox_highThreshold->setMinimum(0);//边缘发现高阈值最小值
    ui->spinBox_LowThreshold->setMaximum(1000);//边缘发现低阈值最大值
    ui->spinBox_LowThreshold->setMinimum(0);//边缘发现低阈值最小值
    ui->spinBox_highThreshold->setValue(300);//边缘发现低阈值
    ui->spinBox_LowThreshold ->setValue(150);//边缘发现高阈值
    ui->spinBox_CannyCoreSize->setMinimum(3);//边缘发现卷积核最小值
    ui->spinBox_CannyCoreSize->setMaximum(7);//边缘发现卷积核最大值
    ui->spinBox_NoiseNum->setMaximum(200000);//椒盐噪声最大数量
    ui->spinBox_NoiseNum->setValue(10000);//椒盐噪声默认数量
    ui->spinBox_RemovalSaltGaussian->setMaximum(1);//去除椒盐噪声/高斯模式最大值
    ui->spinBox_RemovalSaltGaussian->setMinimum(0);//去除椒盐噪声/高斯模式最小值
    ui->spinBox_AddSaltGaussian->setMaximum(1);//产生椒盐噪声/高斯模式最大值
    ui->spinBox_AddSaltGaussian->setMinimum(0);//产生椒盐噪声/高斯模式最小值
    ui->spinBox_RemovalCoreSie->setValue(5);//均值去噪或中值去噪卷积核默认大小
    ui->spinBox_RemovalCoreSie->setMinimum(3);//均值去噪或中值去噪卷积核最小值
    ui->spinBox_diameter->setValue(7);//边缘保留滤波空间领域直径默认值
    ui->spinBox_sigmaColor->setMaximum(1000);//边缘保留滤波颜色差异最大值
    ui->spinBox_sigmaColor->setValue(150);//边缘保留滤波颜色差异默认值
    ui->spinBox_sigmaSpace->setValue(10);//边缘保留滤波空间位置差异默认值
    ui->spinBox_LapacianCoreSize->setValue(3);//拉普拉斯算子卷积核默认大小
    ui->spinBox_thresholdFunction->setMaximum(4);//图像阈值化分割方法得最大值
    ui->doubleSpinBox_thresholdValve->setMaximum(255.0);//图像阈值化分割阈值最大值
    ui->doubleSpinBox_thresholdValve->setValue(127);//图像阈值化分割阈值
    ui->spinBox_GlobalThresholdFunction->setMaximum(1);//全阈值计算模式的最大值

}

ImbaProcess::~ImbaProcess()
{
    delete ui;
}
                                                           //数据采集//
/***********************************************************************************************************************************************
 *@brief:  图片浏览按钮触发，开打图像，图像路径永久保持，图像大小自适应
// *@date:   2024.07.12
 *@param:
***********************************************************************************************************************************************/
void ImbaProcess::on_Btn_open_clicked()
{
    //配置文件完整路径
    QString config_path = qApp->applicationDirPath() + "/config/Setting.ini";
    //通过QSetting类创建配置ini格式文件路径
    std::unique_ptr<QSettings> pIniSet(new QSettings(config_path, QSettings::IniFormat));
    //将配置文件中的值加载进来转换为QString类型存储在LastImagePath中
    QString lastPath = pIniSet->value("/LastImagePath/path").toString();

    if(lastPath.isEmpty())
    {
        //系统标准的图片存储路径
        lastPath = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    }

    QString fileName = QFileDialog::getOpenFileName(this, "请选择图片", lastPath, "图片(*.png *.jpg);;");

    if(fileName.isEmpty())
    {
        //加载失败，向信息区中写入信息
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_Infor->append(QString("本地图片数据采集:%1，图像加载失败，文件路径为空").arg(formattedTime));
        ui->Lab_Image->clear();
        ui->FilePatch->clear();
        return;
    }
    //将路径加载至文件路径显示栏中
    ui->FilePatch->setText(fileName);
    QPixmap *pix = new QPixmap(fileName);
    pix->scaled(ui->Lab_Image->size(),Qt::KeepAspectRatio);
    ui->Lab_Image->setScaledContents(true);
    ui->Lab_Image->setPixmap(*pix);
    //加载成功，向信息区中写入信息
    QDateTime currentDateTime = QDateTime::currentDateTime();
    QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
    ui->textEdit_Infor->append(QString("本地图片数据采集:%1，图像加载成功").arg(formattedTime));
    delete pix;
    pix = nullptr;
    //将文件路径中最后一个斜杠的位置找到，截取最后一个斜杠前面的路径放入INI文件中去
    int end = fileName.lastIndexOf("/");
    QString _path = fileName.left(end);
    pIniSet->setValue("/LastImagePath/path", _path);

}

/********************************************************************************
 *@brief:  开始采集数据，是本地文件数据采集和相机数据采集的功能按钮
 *@date:   2024.07.15
 *@param:
*********************************************************************************/
void ImbaProcess::on_Btn_StarCollect_clicked()
{
 //如果是文件选择框被选中
 if(ui->checkBox_Localfile->checkState()==Qt::Checked){
     BasicImageGlobel::shouldStop=true;
     //使用OPencv函数加载视频加载视频
     cv::VideoCapture Video;
     Video.open(BasicImageGlobel::localFilePath.toStdString(),cv::VideoCaptureAPIs::CAP_FFMPEG);//加载视频的文件路径，和视频格式
     int height =Video.get(cv::CAP_PROP_FRAME_HEIGHT);//视频高
     int width  =Video.get(cv::CAP_PROP_FRAME_WIDTH);//视频宽
     double fps =Video.get(cv::CAP_PROP_FPS);//视频帧率
     double count =Video.get(cv::CAP_PROP_FRAME_COUNT);//总帧率
     //在信息区显示当前加载视频的信息格式
     QDateTime currentDateTime = QDateTime::currentDateTime();
     QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
     ui->textEdit_Infor->append(QString("本地视频数据采集:%1，Height:%2,Width:%3,FPS:%4,Count:%5").arg(formattedTime).arg(height).arg(width).arg(fps).arg(count));
     //视频保存
//         QString WritefileName;
//         cv::VideoWriter Writer(WritefileName.toStdString(),Video.get(cv::CAP_PROP_FOURCC),fps,cv::Size(width,height));
     //将加载的视频写进Qlabel中
     cv::Mat Fram;
     BasicImageGlobel::shouldStop=true;
     while(BasicImageGlobel::shouldStop){
       //读帧
       bool ret = Video.read(Fram);
       if (!ret){
           break;
       }
       //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
       QPixmap pix = matToQPixmap(Fram);
       pix.scaled(ui->label_Camera->size(),Qt::KeepAspectRatio);
       ui->label_Camera->setScaledContents(true);
       ui->label_Camera->setPixmap(pix);
       //检测到停止采集键键直接退出
       if(!BasicImageGlobel::shouldStop){
          break;
       }
       cv::waitKey(100);
     }
     ui->label_Camera->clear();
     ui->checkBox_Localfile->setCheckState(Qt::Unchecked);
     Video.release();
//   Writer.release();
     cv::waitKey(0);
     }
//相机选择框被选中
 else if (ui->checkBox_Camera->checkState()==Qt::Checked){
     //使用OPencv相机数据
     cv::VideoCapture Video;
     Video.open(ui->spinBox_CameraNum->value());//摄像头编号索引为0，加载方式为ANY
     int height =Video.get(cv::CAP_PROP_FRAME_HEIGHT);//视频高
     int width  =Video.get(cv::CAP_PROP_FRAME_WIDTH);//视频宽
     double fps =Video.get(cv::CAP_PROP_FPS);//视频帧率
     double count =Video.get(cv::CAP_PROP_FRAME_COUNT);//总帧率
     //在信息区显示当前加载视频的信息格式
     QDateTime currentDateTime = QDateTime::currentDateTime();
     QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
     ui->textEdit_Infor->append(QString("实时相机数据采集:%1，Height:%2,Width:%3,FPS:%4,Count:%5").arg(formattedTime).arg(height).arg(width).arg(fps).arg(count));
     //视频保存
//         QString WritefileName;
//         cv::VideoWriter Writer(WritefileName.toStdString(),Video.get(cv::CAP_PROP_FOURCC),fps,cv::Size(width,height));
     //将加载的视频写进Qlabel中
     cv::Mat Fram;
     BasicImageGlobel::shouldStop=true;
     while(BasicImageGlobel::shouldStop){
       //读帧
       bool ret = Video.read(Fram);
       if (!ret){
           break;
       }
       //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
       QPixmap pix = matToQPixmap(Fram);
       pix.scaled(ui->label_Camera->size(),Qt::KeepAspectRatio);
       ui->label_Camera->setScaledContents(true);
       ui->label_Camera->setPixmap(pix);
       //检测到停止采集键键直接退出
       if(!BasicImageGlobel::shouldStop){
          break;
       }
       cv::waitKey(100);
     }
     ui->label_Camera->clear();
     Video.release();
//         Writer.release();
     cv::waitKey(0);
     }
}

/********************************************************************************
 *@brief:  停止采集数据，清空视频区显示，并向信息区报告状态和行为
 *@date:   2024.07.15
 *@param:
********************************************************************************/
void ImbaProcess::on_Btn_StopCollect_clicked()
{
   BasicImageGlobel::shouldStop=false;//停止采集
   ui->checkBox_Localfile->setCheckState(Qt::Unchecked);
   ui->checkBox_Camera->setCheckState(Qt::Unchecked);
   //停止数据采集，向信息区中写入信息
   QDateTime currentDateTime = QDateTime::currentDateTime();
   QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
   ui->textEdit_Infor->append(QString("本地图片数据采集:%1，数据停止采集").arg(formattedTime));
}

/********************************************************************************
 *@brief:  选择本地文件 check box
 *@date:   2024.07.15
 *@param:
********************************************************************************/
void ImbaProcess::on_checkBox_Localfile_stateChanged(int state)
{
    //如果check box被选中
    if (state==Qt::Checked){
           ui->checkBox_Camera->setDisabled(true);
           QString lastVideoPath = QStandardPaths::writableLocation(QStandardPaths::MoviesLocation);
           BasicImageGlobel::localFilePath = QFileDialog::getOpenFileName(this, "请选择视频", lastVideoPath, "视频(*.avi *.mp4);;");
           if(BasicImageGlobel::localFilePath.isEmpty())
           {
               //停止数据采集，向信息区中写入信息
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("本地视频数据采集:%1，视频数据采集失败，路径为空").arg(formattedTime));
               ui->checkBox_Localfile->setCheckState(Qt::Unchecked);
               return;
           }
    }
    //禁用相机选项按钮
    else if(state==Qt::Unchecked){
          ui->checkBox_Camera->setDisabled(false);
    }

}
/********************************************************************************
 *@brief:  选择相机数据 check box
 *@date:   2024.07.15
 *@param:
********************************************************************************/
void ImbaProcess::on_checkBox_Camera_stateChanged(int state)
{
    //如果check box被选中
    if (state==Qt::Checked){
           ui->checkBox_Localfile->setDisabled(true);
    }
    //禁用本地文件选项按钮
    else if(state==Qt::Unchecked){
          ui->checkBox_Localfile->setDisabled(false);
    }
}
                                                                //图像直方图//
/***********************************************************************************************************************************************
*@brief: 直方图的绘制与计算，直方图的均衡化，直方图的反向投影
// *@date:   2024.07.16
*@param:
***********************************************************************************************************************************************/
void ImbaProcess::on_Btn_histogram_use_clicked()
{
   if(ui->checkBox_calculate->checkState()==Qt::Checked){//是否进行直方图计算与绘制
       //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
       if(ui->checkBox_imageVido->checkState()==Qt::Unchecked){
         cv::Mat image;
         image=cv::imread(ui->FilePatch->text().toStdString());
         if(image.empty()){
             QDateTime currentDateTime = QDateTime::currentDateTime();
             QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
             ui->textEdit_Infor->append(QString("本地图片数据应用:%1，直方图计算失败，文件路径为空").arg(formattedTime));
             BasicImageGlobel::histogramSucc=false;//直方图计算失败
             ui->Lab_Image->clear();
            return;
         }
         //计算图像直方图
         BasicImageGlobel::histData=QPixmapToMat(calcHistogram(image));//直方图数据
         BasicImageGlobel::histimage=image;//直方图对应的原图
         ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::histData).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
         BasicImageGlobel::histogramSucc=true;//直方图计算成功
         QDateTime currentDateTime = QDateTime::currentDateTime();
         QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
         ui->textEdit_Infor->append(QString("本地图片数据应用:%1，直方图计算并绘制成功").arg(formattedTime));
      }
       //计算视频直方图
       else if (ui->checkBox_imageVido->checkState()==Qt::Checked){
          //先判断当前视频Label
          if(ui->label_Camera->pixmap()!=nullptr ){
            const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
            BasicImageGlobel::histimage=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
            QPixmap HistogramPixmap =calcHistogram(BasicImageGlobel::histimage);//计算图像直方图
            BasicImageGlobel::histData=QPixmapToMat(HistogramPixmap);//直方图数据Mat对象
            ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::histData).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
            //打印成功时间
            QDateTime currentDateTime = QDateTime::currentDateTime();
            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
            ui->textEdit_Infor->append(QString("本地视频数据应用:%1，直方图计算并绘制成功").arg(formattedTime));
          }
          else{
            //打印失败时间
            QDateTime currentDateTime = QDateTime::currentDateTime();
            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
            ui->textEdit_Infor->append(QString("本地图片数据应用:%1，直方图计算失败，视频为空").arg(formattedTime));
          }
       }
   }

    //直方图的均衡化(均衡化之前，首先得计算好了直方图，这里是直接取出的直方图数据，这里就不区分是视频数据还是本地图片数据了，因为首先要计算直方图数据，才能够进行均衡化)
    if(!BasicImageGlobel::histData.empty() && !BasicImageGlobel::histimage.empty() && ui->checkBox_equalization->checkState()==Qt::Checked){//直方图数据和直方图原图不为空，且选中直方图均衡化的选型
        if(ui->checkBox_Global_Part->checkState()==Qt::Checked){//局部自适应均衡化
          cv::Mat dstimage,Result;
          cvtColor(BasicImageGlobel::histimage,dstimage,cv::COLOR_BGR2HSV);
          //通道分离
          std::vector<cv::Mat>mv;
          split(dstimage,mv);
          //对亮度空间进行局部均衡化
          auto Clahe=createCLAHE(ui->doubleSpinBox_contrast->value(),cv::Size(8,8));
          Clahe->apply(mv[ui->spinBox_HSVchannel->value()],mv[ui->spinBox_HSVchannel->value()]);
          //通道合并
          merge(mv,Result);
          cvtColor(Result,BasicImageGlobel::PartequalizeHistimge,cv::COLOR_HSV2BGR);
          //合并后的图像计算直方图并输出
          BasicImageGlobel::PartequalizeHistData=QPixmapToMat(calcHistogram(Result));
          ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::PartequalizeHistData).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
          QDateTime currentDateTime = QDateTime::currentDateTime();
          QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
          ui->textEdit_Infor->append(QString("本地图片数据均衡化应用:%1，局部直方图均衡化计算并绘制成功").arg(formattedTime));
        }
        else if(ui->checkBox_Global_Part->checkState()==Qt::Unchecked){//全局均衡化
               //判断图像是否通道数大于1，不是灰度图像，那么就将图像转换为灰度图像
               cv::Mat Grayimage;
              if(BasicImageGlobel::histimage.channels()>1){
                cv::cvtColor(BasicImageGlobel::histimage,Grayimage,cv::COLOR_BGR2GRAY);//将图像通道转换为单通道
              }
              //执行全局阈值化
              cv::equalizeHist(Grayimage,BasicImageGlobel::GlobelequalizeHistimge);
              //将全局阈值化的直方图数据计算出来
              BasicImageGlobel::GlobelequalizeHistData=QPixmapToMat(calcHistogram(BasicImageGlobel::GlobelequalizeHistimge));
              ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::GlobelequalizeHistData).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
              QDateTime currentDateTime = QDateTime::currentDateTime();
              QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
              ui->textEdit_Infor->append(QString("本地图片数据均衡化应用:%1，全局直方图均衡化计算并绘制成功").arg(formattedTime));
        }
    }
    else if((BasicImageGlobel::histData.empty()||BasicImageGlobel::histimage.empty()) && ui->checkBox_equalization->checkState()==Qt::Checked){//直方图数据为空 或者直方图原图为空，且选中直方图均衡化的选型报警
        //打印失败时间
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_Infor->append(QString("本地图片数据应用:%1，直方图均衡化失败，直方图数据为空，请先计算直方图数据").arg(formattedTime));
    }
    //直方图反向投影，首先得计算好了直方图，这里是直接取出的直方图数据，当然这里就不区分是视频数据还是本地图片数据了，因为首先要计算直方图数据，才能够进行均衡化)


}

//展示直方图
void ImbaProcess::on_Btn_ShowHistogram_clicked()
{
    //如果直方图数据不为空，且直方图均衡化未选中，且直方图反向投影未选中，那么就直接显示计算的直方图数据
    if(!BasicImageGlobel::histData.empty()&&ui->checkBox_equalization->checkState()==Qt::Unchecked&&ui->checkBox_backProjection->checkState()==Qt::Unchecked){
        ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::histData.clone()).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
    }
    //直方图均衡化选中，那么就显示直方图均衡化后的结果
    else if(ui->checkBox_equalization->checkState()==Qt::Checked&&ui->checkBox_backProjection->checkState()==Qt::Unchecked){
             //展示局部阈值均衡化直方图数据 局部直方图均衡化中的数据不为空
        if(ui->checkBox_Global_Part->checkState()==Qt::Checked&&!BasicImageGlobel::PartequalizeHistData.empty()){
            ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::PartequalizeHistData.clone()).scaled(ui->Lab_Image->size()));//计算局部阈值直方图数据后返回并设置到Qlabel中
        }
             //展示全局阈值均衡化直方图数据 全局直方图均衡化中的数据不为空
        else if(ui->checkBox_Global_Part->checkState()==Qt::Unchecked&&!BasicImageGlobel::GlobelequalizeHistData.empty()){
            ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::GlobelequalizeHistData.clone()).scaled(ui->Lab_Image->size()));//计算全局阈值直方图数据后返回并设置到Qlabel中
        }
    }
    else{
     //打印失败时间
     QDateTime currentDateTime = QDateTime::currentDateTime();
     QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
     ui->textEdit_Infor->append(QString("展示直方图数据:%1，直方图展示失败，数据为空").arg(formattedTime));
    }
}

//展示原图
void ImbaProcess::on_Btn_Showimage_clicked()
{
    //如果直方图原图不为空，且直方图均衡化未选中，且直方图反向投影未选中，那么就直接显示计算的直方图原图
    if(!BasicImageGlobel::histimage.empty() && ui->checkBox_equalization->checkState()==Qt::Unchecked&&ui->checkBox_backProjection->checkState()==Qt::Unchecked){
          ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::histimage.clone()).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
    }
    else if(ui->checkBox_equalization->checkState()==Qt::Checked&&ui->checkBox_backProjection->checkState()==Qt::Unchecked){
        //展示局部阈值均衡化直方图原图 局部直方图原图中的数据不为空
        if(ui->checkBox_Global_Part->checkState()==Qt::Checked&&!BasicImageGlobel::PartequalizeHistimge.empty()){
             ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::PartequalizeHistimge.clone()).scaled(ui->Lab_Image->size()));//计算局部阈值原图数据后返回并设置到Qlabel中
         }
        //展示全局阈值均衡化直方图原图 全局直方图原图中的数据不为空
        else if(ui->checkBox_Global_Part->checkState()==Qt::Unchecked&&!BasicImageGlobel::GlobelequalizeHistimge.empty()){
            ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::GlobelequalizeHistimge.clone()).scaled(ui->Lab_Image->size()));//计算全局阈值原图数据后返回并设置到Qlabel中
        }
    }
    else{
     //打印失败时间
     QDateTime currentDateTime = QDateTime::currentDateTime();
     QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
     ui->textEdit_Infor->append(QString("展示原图数据:%1，原图展示失败，数据为空").arg(formattedTime));
    }
}


                                                               //图像卷积//
/***********************************************************************************************************************************************
*@brief: 卷积模糊，自定义滤波，梯度提取，边缘发现，噪声与去噪，边缘保留滤波，锐化增强
// *@date:   2024.07.16
*@param:
***********************************************************************************************************************************************/
//卷积模糊
void ImbaProcess::on_checkBox_ConvolutionaAmbiguity_stateChanged(int state)
{
    //卷积模糊选中计算
   if(state==Qt::Checked){
       //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
       if(ui->checkBox_ConvolutionImageVideo->checkState()==Qt::Unchecked){
           cv::Mat image;
           image=cv::imread(ui->FilePatch->text().toStdString());
           if(image.empty()){
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("本地图片数据应用:%1，卷积模糊失败，文件路径为空").arg(formattedTime));
               ui->Lab_Image->clear();
              return;
           }
           //均值模糊
           if(ui->spinBox_Convolutionamode->value()==0){
               //判断卷积核是否是奇数，否则加1再进行计算
               if(ui->spinBox_ScoreSize->value()%2==0){
                  ui->spinBox_ScoreSize->setValue(ui->spinBox_ScoreSize->value()+1);
               }
               //均值函数
               cv::blur(image,BasicImageGlobel::ConvolutionaAmbiguityResult,cv::Size(ui->spinBox_ScoreSize->value(),ui->spinBox_ScoreSize->value()),cv::Point(-1,-1),ui->spinBox_ConvolutionaBorder->value());
               ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::ConvolutionaAmbiguityResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("本地图片数据卷积结果:%1，均值模糊计算并绘制成功").arg(formattedTime));
           }
           //高斯模糊
           else if(ui->spinBox_Convolutionamode->value()==1){
               //判断卷积核是否是奇数，否则加1再进行计算
               if(ui->spinBox_ScoreSize->value()%2==0){
                  ui->spinBox_ScoreSize->setValue(ui->spinBox_ScoreSize->value()+1);
               }
               //高斯函数
               cv::GaussianBlur(image,BasicImageGlobel::ConvolutionaAmbiguityResult,cv::Size(ui->spinBox_ScoreSize->value(),ui->spinBox_ScoreSize->value()),0,0,ui->spinBox_ConvolutionaBorder->value());
               ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::ConvolutionaAmbiguityResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("本地图片数据卷积结果:%1，高斯模糊计算并绘制成功").arg(formattedTime));

           }
           //中值模糊
           else if(ui->spinBox_Convolutionamode->value()==2){
               //判断卷积核是否是奇数，否则加1再进行计算
               if(ui->spinBox_ScoreSize->value()%2==0){
                  ui->spinBox_ScoreSize->setValue(ui->spinBox_ScoreSize->value()+1);
               }
               //中值模糊函数
               cv::medianBlur(image,BasicImageGlobel::ConvolutionaAmbiguityResult,ui->spinBox_ScoreSize->value());
               ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::ConvolutionaAmbiguityResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("本地图片数据卷积结果:%1，中模糊计算并绘制成功").arg(formattedTime));
           }
       }
       //计算视频卷积模糊
       else if (ui->checkBox_ConvolutionImageVideo->checkState()==Qt::Checked){
           //先判断当前视频Label
           if(ui->label_Camera->pixmap()!=nullptr ){
             const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
             BasicImageGlobel::histimage=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
           }                    
           //均值模糊
           if(ui->spinBox_Convolutionamode->value()==0){
               //判断卷积核是否是奇数，否则加1再进行计算
               if(ui->spinBox_ScoreSize->value()%2==0){
                  ui->spinBox_ScoreSize->setValue(ui->spinBox_ScoreSize->value()+1);
               }
               //均值函数
               cv::blur(BasicImageGlobel::histimage,BasicImageGlobel::ConvolutionaAmbiguityResult,cv::Size(ui->spinBox_ScoreSize->value(),ui->spinBox_ScoreSize->value()),cv::Point(-1,-1),ui->spinBox_ConvolutionaBorder->value());
               ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::ConvolutionaAmbiguityResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("视频数据卷积结果:%1，均值模糊计算并绘制成功").arg(formattedTime));
           }
           //高斯模糊
           else if(ui->spinBox_Convolutionamode->value()==1){
               //判断卷积核是否是奇数，否则加1再进行计算
               if(ui->spinBox_ScoreSize->value()%2==0){
                  ui->spinBox_ScoreSize->setValue(ui->spinBox_ScoreSize->value()+1);
               }
               //高斯函数
               cv::GaussianBlur(BasicImageGlobel::histimage,BasicImageGlobel::ConvolutionaAmbiguityResult,cv::Size(ui->spinBox_ScoreSize->value(),ui->spinBox_ScoreSize->value()),0,0,ui->spinBox_ConvolutionaBorder->value());
               ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::ConvolutionaAmbiguityResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("视频数据卷积结果:%1，高斯模糊计算并绘制成功").arg(formattedTime));

           }
           //中值模糊
           else if(ui->spinBox_Convolutionamode->value()==2){
               //判断卷积核是否是奇数，否则加1再进行计算
               if(ui->spinBox_ScoreSize->value()%2==0){
                  ui->spinBox_ScoreSize->setValue(ui->spinBox_ScoreSize->value()+1);
               }
               //中值模糊函数
               cv::medianBlur(BasicImageGlobel::histimage,BasicImageGlobel::ConvolutionaAmbiguityResult,ui->spinBox_ScoreSize->value());
               ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::ConvolutionaAmbiguityResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("视频数据卷积结果:%1，中模糊计算并绘制成功").arg(formattedTime));
           }
       }
   }
}
//自定义滤波
void ImbaProcess::on_checkBox_CustomFiltering_stateChanged(int state){
    //自定义滤波选中计算
   if(state==Qt::Checked){
       //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
       if(ui->checkBox_ConvolutionImageVideo->checkState()==Qt::Unchecked){
           cv::Mat image;
           image=cv::imread(ui->FilePatch->text().toStdString());
           if(image.empty()){
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("本地图片数据应用:%1，卷积模糊失败，文件路径为空").arg(formattedTime));
               ui->Lab_Image->clear();
              return;
           }
             //水平模糊
             if(ui->spinBox_VH->value()==0){
                cv::Mat Kernel=cv::Mat::ones(cv::Size(ui->spinBox_specificValue->value(),1),CV_32FC1);//水平核
                Kernel=Kernel/25;
                cv::filter2D(image,BasicImageGlobel::CustomFiltering,-1,Kernel,cv::Point(-1,-1));
                ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::CustomFiltering).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
                QDateTime currentDateTime = QDateTime::currentDateTime();
                QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                ui->textEdit_Infor->append(QString("本地数据自定义水平模糊结果:%1，自定义水平模糊计算并绘制成功").arg(formattedTime));
                }
             //垂直模糊
             else if(ui->spinBox_VH->value()==1){
                 cv::Mat Kernel=cv::Mat::ones(cv::Size(1,ui->spinBox_specificValue->value()),CV_32FC1);//垂直核
                 Kernel=Kernel/25;
                 cv::filter2D(image,BasicImageGlobel::CustomFiltering,-1,Kernel,cv::Point(-1,-1));
                 ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::CustomFiltering).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
                 QDateTime currentDateTime = QDateTime::currentDateTime();
                 QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                 ui->textEdit_Infor->append(QString("本地数据自定义垂直模糊结果:%1，自定义垂直模糊计算并绘制成功").arg(formattedTime));
                 }
          }
           //计算视频自定义滤波
           else if (ui->checkBox_ConvolutionImageVideo->checkState()==Qt::Checked){
               //先判断当前视频Label
               if(ui->label_Camera->pixmap()!=nullptr ){
                 const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
                 BasicImageGlobel::histimage=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
               }
               //水平模糊
               if(ui->spinBox_VH->value()==0){
                   cv::Mat Kernel=cv::Mat::ones(cv::Size(ui->spinBox_specificValue->value(),1),CV_32FC1);//水平核
                   cv::filter2D(BasicImageGlobel::histimage,BasicImageGlobel::CustomFiltering,-1,Kernel,cv::Point(-1,-1));
                   ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::CustomFiltering).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
                   QDateTime currentDateTime = QDateTime::currentDateTime();
                   QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                   ui->textEdit_Infor->append(QString("视频数据自定义水平模糊结果:%1，自定义水平模糊计算并绘制成功").arg(formattedTime));
               }
               //垂直模糊
               else if(ui->spinBox_VH->value()==1){
                   cv::Mat Kernel=cv::Mat::ones(cv::Size(1,ui->spinBox_specificValue->value()),CV_32FC1);//垂直核
                   cv::filter2D(BasicImageGlobel::histimage,BasicImageGlobel::CustomFiltering,-1,Kernel,cv::Point(-1,-1));
                   ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::CustomFiltering).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
                   QDateTime currentDateTime = QDateTime::currentDateTime();
                   QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                   ui->textEdit_Infor->append(QString("视频数据自定义垂直模糊结果:%1，自定义垂直模糊计算并绘制成功").arg(formattedTime));
                   }
             }
   }
}
//梯度提取
void ImbaProcess::on_checkBox_GradientExtraction_stateChanged(int state){
    //梯度提取选中计算
   if(state==Qt::Checked){
       //判断卷积核是否是奇数，否则加1再进行计算
       if(ui->spinBox_SobelCoreSize->value()%2==0){
          ui->spinBox_SobelCoreSize->setValue(ui->spinBox_SobelCoreSize->value()+1);
       }
       //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
       if(ui->checkBox_ConvolutionImageVideo->checkState()==Qt::Unchecked){
           cv::Mat image;
           image=cv::imread(ui->FilePatch->text().toStdString());
           if(image.empty()){
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("本地图片数据应用:%1，梯度提取失败，文件路径为空").arg(formattedTime));
               ui->Lab_Image->clear();
               return;
              }
              if(ui->spinBox_gradientXY->value()==0){//X方向梯度
                  if(ui->spinBox_SobelScharr->value()==0){//使用Sobel
                      cv::Sobel(image,BasicImageGlobel::gradientResult,CV_32F,1,0,ui->spinBox_SobelCoreSize->value());//使用Sobel计算X方向梯度
                  }else if(ui->spinBox_SobelScharr->value()==1){//使用Scharr计算X方向梯度
                      cv::Scharr(image,BasicImageGlobel::gradientResult,CV_32F,1,0);
                  }
                   //归一化
                   cv::normalize(BasicImageGlobel::gradientResult,BasicImageGlobel::gradientResult,1,0,cv::NORM_MINMAX);
                   BasicImageGlobel::gradientResult.convertTo(BasicImageGlobel::gradientResult,CV_8U,255.0);//将CV_32F得图像转换为CV_8U
              }else if(ui->spinBox_gradientXY->value()==1){//Y方向梯度
                  if(ui->spinBox_SobelScharr->value()==0){//使用Sobel
                      cv::Sobel(image,BasicImageGlobel::gradientResult,CV_32F,0,1,ui->spinBox_SobelCoreSize->value());//使用Sobel计算Y方向梯度
                   }else if(ui->spinBox_SobelScharr->value()==1){//使用Scharr计算Y方向梯度
                      cv::Scharr(image,BasicImageGlobel::gradientResult,CV_32F,0,1);
                  }
                   //归一化
                   cv::normalize(BasicImageGlobel::gradientResult,BasicImageGlobel::gradientResult,1,0,cv::NORM_MINMAX);
                   BasicImageGlobel::gradientResult.convertTo(BasicImageGlobel::gradientResult,CV_8U,255.0);//将CV_32F得图像转换为CV_8U
              }
              ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::gradientResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
              QDateTime currentDateTime = QDateTime::currentDateTime();
              QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
              ui->textEdit_Infor->append(QString("本地数据梯度提取结果:%1，梯度提取计算并绘制成功").arg(formattedTime));
            }

       //计算视频梯度
       else if (ui->checkBox_ConvolutionImageVideo->checkState()==Qt::Checked){
           //先判断当前视频Label
           if(ui->label_Camera->pixmap()!=nullptr ){
             const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
             BasicImageGlobel::histimage=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
           }

           if(ui->spinBox_gradientXY->value()==0){//X方向梯度
               if(ui->spinBox_SobelScharr->value()==0){//使用Sobel
                   cv::Sobel(BasicImageGlobel::histimage,BasicImageGlobel::gradientResult,CV_32F,1,0,ui->spinBox_SobelCoreSize->value());//使用Sobel计算X方向梯度
               }else if(ui->spinBox_SobelScharr->value()==1){//使用Scharr计算X方向梯度
                   cv::Scharr(BasicImageGlobel::histimage,BasicImageGlobel::gradientResult,CV_32F,1,0);
               }
                //归一化
                cv::normalize(BasicImageGlobel::gradientResult,BasicImageGlobel::gradientResult,1,0,cv::NORM_MINMAX);
                BasicImageGlobel::gradientResult.convertTo(BasicImageGlobel::gradientResult,CV_8U,255.0);//将CV_32F得图像转换为CV_8U
           }else if(ui->spinBox_gradientXY->value()==1){//Y方向梯度
               if(ui->spinBox_SobelScharr->value()==0){//使用Sobel
                   cv::Sobel(BasicImageGlobel::histimage,BasicImageGlobel::gradientResult,CV_32F,0,1,ui->spinBox_SobelCoreSize->value());//使用Sobel计算Y方向梯度
               }else if(ui->spinBox_SobelScharr->value()==1){//使用Scharr计算Y方向梯度
                   cv::Scharr(BasicImageGlobel::histimage,BasicImageGlobel::gradientResult,CV_32F,0,1);
               }
                //归一化
                cv::normalize(BasicImageGlobel::gradientResult,BasicImageGlobel::gradientResult,1,0,cv::NORM_MINMAX);
                BasicImageGlobel::gradientResult.convertTo(BasicImageGlobel::gradientResult,CV_8U,255.0);//将CV_32F得图像转换为CV_8U
           }
           ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::gradientResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
           QDateTime currentDateTime = QDateTime::currentDateTime();
           QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
           ui->textEdit_Infor->append(QString("本地数据梯度提取结果:%1，梯度提取计算并绘制成功").arg(formattedTime));
         }
     }
}
//边缘发现
void ImbaProcess::on_checkBox_EdgeDiscovery_stateChanged(int state)
{
    //边缘发现选中计算
   if(state==Qt::Checked){
       //判断卷积核是否是奇数，否则加1再进行计算
       if(ui->spinBox_CannyCoreSize->value()%2==0){
          ui->spinBox_CannyCoreSize->setValue(ui->spinBox_CannyCoreSize->value()+1);
       }
       //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
       if(ui->checkBox_ConvolutionImageVideo->checkState()==Qt::Unchecked){
           cv::Mat image;
           image=cv::imread(ui->FilePatch->text().toStdString());
           if(image.empty()){
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("本地图片数据应用:%1，边缘发现失败，文件路径为空").arg(formattedTime));
               ui->Lab_Image->clear();
               return;
              }
              //边缘发现
              cv::Canny(image,BasicImageGlobel::CannyResult,ui->spinBox_LowThreshold->value(),ui->spinBox_highThreshold->value(),ui->spinBox_CannyCoreSize->value());
              ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::CannyResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
              QDateTime currentDateTime = QDateTime::currentDateTime();
              QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
              ui->textEdit_Infor->append(QString("本地数据边缘发现结果:%1，边缘发现计算并绘制成功").arg(formattedTime));
            }
       else if (ui->checkBox_ConvolutionImageVideo->checkState()==Qt::Checked){
           //先判断当前视频Label
           if(ui->label_Camera->pixmap()!=nullptr ){
             const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
             BasicImageGlobel::histimage=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
           }
           //边缘发现
           cv::Canny(BasicImageGlobel::histimage,BasicImageGlobel::CannyResult,ui->spinBox_LowThreshold->value(),ui->spinBox_highThreshold->value(),ui->spinBox_CannyCoreSize->value());
           ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::CannyResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
           QDateTime currentDateTime = QDateTime::currentDateTime();
           QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
           ui->textEdit_Infor->append(QString("视频数据边缘发现结果:%1，边缘发现计算并绘制成功").arg(formattedTime));
         }
     }
}
//噪声与去噪
void ImbaProcess::on_checkBox_NoiseDenoising_stateChanged(int state)
{
    //噪声与去噪选中计算
   if(state==Qt::Checked){
       //判断卷积核是否是奇数，否则加1再进行计算
       if(ui->spinBox_RemovalCoreSie->value()%2==0){
          ui->spinBox_RemovalCoreSie->setValue(ui->spinBox_RemovalCoreSie->value()+1);
       }
       //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
       if(ui->checkBox_ConvolutionImageVideo->checkState()==Qt::Unchecked){
           cv::Mat image;
           image=cv::imread(ui->FilePatch->text().toStdString());
           if(image.empty()){
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("本地图片数据应用:%1，梯度提取失败，文件路径为空").arg(formattedTime));
               ui->Lab_Image->clear();
               return;
              }
              //去除还是产生噪声判断
              if(ui->checkBox_NoiseAddRemoval->checkState()==Qt::Unchecked){//去除噪声
                 if(ui->spinBox_RemovalSaltGaussian->value()==0){//中值去噪
                    cv::medianBlur(image,BasicImageGlobel::Noiseprocess,ui->spinBox_RemovalCoreSie->value());
                 }else if(ui->spinBox_RemovalSaltGaussian->value()==1){//均值去噪
                    cv::blur(image,BasicImageGlobel::Noiseprocess,cv::Size(ui->spinBox_RemovalCoreSie->value(),ui->spinBox_RemovalCoreSie->value()));
                 }
                 ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::Noiseprocess).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
                 QDateTime currentDateTime = QDateTime::currentDateTime();
                 QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                 ui->textEdit_Infor->append(QString("本地数据噪声去除结果:%1，噪声去除并绘制成功").arg(formattedTime));
              }else if(ui->checkBox_NoiseAddRemoval->checkState()==Qt::Checked){//产生噪声
                  if(ui->spinBox_AddSaltGaussian->value()==0){//产生椒盐
                     cv::RNG rng(QTime::currentTime().toString().toInt());
                     int height=image.rows;
                     int width =image.cols;
                     BasicImageGlobel::Noiseprocess=image.clone();
                     for (int i=0;i<ui->spinBox_NoiseNum->value();i++) {
                          int x=rng.uniform(0,width);
                          int y=rng.uniform(0,height);
                          if(i%2==0){
                              BasicImageGlobel::Noiseprocess.at<cv::Vec3b>(y,x)=cv::Vec3b(255,255,255);//黑点
                          }else{
                              BasicImageGlobel::Noiseprocess.at<cv::Vec3b>(y,x)=cv::Vec3b(0,0,0);//白点
                          }
                     }

                  }else if(ui->spinBox_AddSaltGaussian->value()==1){//产生高斯
                           cv::Mat noise =cv::Mat::zeros(image.size(),image.type());
                           cv::randn(noise,(15,15,15),(30,30,30));
                           cv::add(image,noise,BasicImageGlobel::Noiseprocess);
                  }
                  ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::Noiseprocess).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
                  QDateTime currentDateTime = QDateTime::currentDateTime();
                  QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                  ui->textEdit_Infor->append(QString("本地数据噪声产生结果:%1，噪声产生并绘制成功").arg(formattedTime));
              }
            }
       else if (ui->checkBox_ConvolutionImageVideo->checkState()==Qt::Checked){
           //先判断当前视频Label
           if(ui->label_Camera->pixmap()!=nullptr ){
             const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
             BasicImageGlobel::histimage=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
           }
           //去除还是产生噪声判断
           if(ui->checkBox_NoiseAddRemoval->checkState()==Qt::Unchecked){//去除噪声
              if(ui->spinBox_RemovalSaltGaussian->value()==0){//中值去噪
                 cv::medianBlur(BasicImageGlobel::histimage,BasicImageGlobel::Noiseprocess,ui->spinBox_RemovalCoreSie->value());
              }else if(ui->spinBox_RemovalSaltGaussian->value()==1){//均值去噪
                 cv::blur(BasicImageGlobel::histimage,BasicImageGlobel::Noiseprocess,cv::Size(ui->spinBox_RemovalCoreSie->value(),ui->spinBox_RemovalCoreSie->value()));
              }
              ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::Noiseprocess).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
              QDateTime currentDateTime = QDateTime::currentDateTime();
              QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
              ui->textEdit_Infor->append(QString("视频数据噪声去除结果:%1，噪声去除并绘制成功").arg(formattedTime));
           }else if(ui->checkBox_NoiseAddRemoval->checkState()==Qt::Checked){//产生噪声
               if(ui->spinBox_AddSaltGaussian->value()==0){//产生椒盐
                  cv::RNG rng(QTime::currentTime().toString().toInt());
                  int height=BasicImageGlobel::histimage.rows;
                  int width =BasicImageGlobel::histimage.cols;
                  BasicImageGlobel::Noiseprocess=BasicImageGlobel::histimage.clone();
                  for (int i=0;i<ui->spinBox_NoiseNum->value();i++) {
                       int x=rng.uniform(0,width);
                       int y=rng.uniform(0,height);
                       if(i%2==0){
                           BasicImageGlobel::Noiseprocess.at<cv::Vec3b>(y,x)=cv::Vec3b(255,255,255);//黑点
                       }else{
                           BasicImageGlobel::Noiseprocess.at<cv::Vec3b>(y,x)=cv::Vec3b(0,0,0);//白点
                       }
                  }

               }else if(ui->spinBox_AddSaltGaussian->value()==1){//产生高斯
                        cv::Mat noise =cv::Mat::zeros(BasicImageGlobel::histimage.size(),BasicImageGlobel::histimage.type());
                        cv::randn(noise,(15,15,15),(30,30,30));
                        cv::add(BasicImageGlobel::histimage,noise,BasicImageGlobel::Noiseprocess);
               }
               ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::Noiseprocess).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("视频数据噪声产生结果:%1，噪声产生并绘制成功").arg(formattedTime));
            }

         }
    }
}
//边缘保留滤波
void ImbaProcess::on_checkBox_EdgeRetentionFiltering_stateChanged(int state){
    //边缘保留滤波选中计算
   if(state==Qt::Checked){
       //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
       if(ui->checkBox_ConvolutionImageVideo->checkState()==Qt::Unchecked){
           cv::Mat image;
           image=cv::imread(ui->FilePatch->text().toStdString());
           if(image.empty()){
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("本地图片数据应用:%1，梯度提取失败，文件路径为空").arg(formattedTime));
               ui->Lab_Image->clear();
               return;
              }
              cv::bilateralFilter(image,BasicImageGlobel::BliateraResult,ui->spinBox_diameter->value(),ui->spinBox_sigmaColor->value(),ui->spinBox_sigmaSpace->value());
              ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::BliateraResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
              QDateTime currentDateTime = QDateTime::currentDateTime();
              QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
              ui->textEdit_Infor->append(QString("本地数据边缘保留滤波结果:%1，高斯双边滤波并绘制成功").arg(formattedTime));

            }
           else if (ui->checkBox_ConvolutionImageVideo->checkState()==Qt::Checked){
               //先判断当前视频Label
               if(ui->label_Camera->pixmap()!=nullptr ){
                 const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
                 BasicImageGlobel::histimage=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
               }
               cv::bilateralFilter(BasicImageGlobel::histimage,BasicImageGlobel::BliateraResult,ui->spinBox_diameter->value(),ui->spinBox_sigmaColor->value(),ui->spinBox_sigmaSpace->value());
               ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::BliateraResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("视频数据边缘保留滤波结果:%1，高斯双边滤波并绘制成功").arg(formattedTime));
           }
     }
}
//锐化增强
void ImbaProcess::on_checkBox_SharpeningEnhancement_stateChanged(int state)
{
    //判断卷积核是否是奇数，否则加1再进行计算
    if(ui->spinBox_LapacianCoreSize->value()%2==0){
       ui->spinBox_LapacianCoreSize->setValue(ui->spinBox_LapacianCoreSize->value()+1);
    }
    //锐化增强选中计算
   if(state==Qt::Checked){
       //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
       if(ui->checkBox_ConvolutionImageVideo->checkState()==Qt::Unchecked){
           cv::Mat image;
           image=cv::imread(ui->FilePatch->text().toStdString());
           if(image.empty()){
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("本地图片数据应用:%1，梯度提取失败，文件路径为空").arg(formattedTime));
               ui->Lab_Image->clear();
               return;
              }
              //拉普拉斯
              cv::Laplacian(image,BasicImageGlobel::SharpenResult,CV_32F,ui->spinBox_LapacianCoreSize->value());
              cv::normalize(BasicImageGlobel::SharpenResult,BasicImageGlobel::SharpenResult,0,255.0,cv::NORM_MINMAX);//归一化
              BasicImageGlobel::SharpenResult.convertTo(BasicImageGlobel::SharpenResult,CV_8U);//转换图像类型
              ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::SharpenResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
              QDateTime currentDateTime = QDateTime::currentDateTime();
              QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
              ui->textEdit_Infor->append(QString("本地数据锐化增强结果:%1，拉普拉斯锐化增强并绘制成功").arg(formattedTime));

            }
           else if (ui->checkBox_ConvolutionImageVideo->checkState()==Qt::Checked){
               //先判断当前视频Label
               if(ui->label_Camera->pixmap()!=nullptr ){
                 const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
                 BasicImageGlobel::histimage=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
               }
               //拉普拉斯
               cv::Laplacian(BasicImageGlobel::histimage,BasicImageGlobel::SharpenResult,CV_32F,ui->spinBox_LapacianCoreSize->value());
               cv::normalize(BasicImageGlobel::SharpenResult,BasicImageGlobel::SharpenResult,0,255.0,cv::NORM_MINMAX);//归一化
               BasicImageGlobel::SharpenResult.convertTo(BasicImageGlobel::SharpenResult,CV_8U);//转换图像类型
               ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::SharpenResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("视频数据拉普拉斯锐化增强结果:%1，拉普拉斯锐化增强并绘制成功").arg(formattedTime));
           }
   }
}
                                                          //二值分析//
/***********************************************************************************************************************************************
*@brief: 图像阈值化分割，全阈值计算，自适应阈值计算，去噪与二值化，联通组件标记，轮廓发现与绘制，拟合与逼近，轮廓分析，霍夫直线检测，霍夫圆检测，
         最大内接圆和最小外接圆，轮廓匹配，最大轮廓与关键点编码，凸包检测
//*@date:   2024.07.30
*@param:
***********************************************************************************************************************************************/
//图像阈值化分割
void ImbaProcess::on_checkBox_thresholdSegmentation_stateChanged(int state){
 //选中则计算
 if(state==Qt::Checked){
    cv::Mat imageVideo;//存储视频或者本地图片中间变量
    //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
    if(ui->checkBox_BinaryImageVideo->checkState()==Qt::Unchecked){
        imageVideo=cv::imread(ui->FilePatch->text().toStdString());
        if(imageVideo.empty()){
            QDateTime currentDateTime = QDateTime::currentDateTime();
            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
            ui->textEdit_Infor->append(QString("本地图片数据应用:%1，阈值化分割失败，文件路径为空").arg(formattedTime));
            ui->Lab_Image->clear();
            return;
           }
      }else if (ui->checkBox_BinaryImageVideo->checkState()==Qt::Checked){
        //先判断当前视频Label
        if(ui->label_Camera->pixmap()!=nullptr ){
          const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
          imageVideo=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
          }
       }
      //图像阈值化分割
      cv::cvtColor(imageVideo,imageVideo,cv::COLOR_BGR2GRAY);
      cv::threshold(imageVideo,BasicImageGlobel::ThresholdResult,ui->doubleSpinBox_thresholdValve->value(),255,ui->spinBox_thresholdFunction->value());
      ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::ThresholdResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
      QDateTime currentDateTime = QDateTime::currentDateTime();
      QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
      ui->textEdit_Infor->append(QString("图像阈值化分割结果:%1，图像阈值化分割并绘制成功").arg(formattedTime));
   }
}

//全阈值计算
void ImbaProcess::on_checkBox_GlobalComputing_stateChanged(int state){
    //选中则计算
    if(state==Qt::Checked){
       cv::Mat imageVideo;//存储视频或者本地图片中间变量
       //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
       if(ui->checkBox_BinaryImageVideo->checkState()==Qt::Unchecked){
           imageVideo=cv::imread(ui->FilePatch->text().toStdString());
           if(imageVideo.empty()){
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("本地图片数据应用:%1，全阈值计算失败，文件路径为空").arg(formattedTime));
               ui->Lab_Image->clear();
               return;
              }
         }else if (ui->checkBox_BinaryImageVideo->checkState()==Qt::Checked){
           //先判断当前视频Label
           if(ui->label_Camera->pixmap()!=nullptr ){
             const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
             imageVideo=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
             }
          }
       //图像阈值化分割
       cv::cvtColor(imageVideo,imageVideo,cv::COLOR_BGR2GRAY);
       int function;
       if(ui->spinBox_GlobalThresholdFunction->value()==0){
          function=8;//大津法 THRESH_OTSU = 8, //!< flag, use Otsu algorithm to choose the optimal threshold value
       }else if(ui->spinBox_GlobalThresholdFunction->value()==1){
          function=16;//三角法 THRESH_TRIANGLE = 16 //!< flag, use Triangle algorithm to choose the optimal threshold value
       }
       //阈值计算并将阈值结果写入显示区
       ui->lineEdit_GlobalComputingRult->setText(QString().setNum(cv::threshold(imageVideo,BasicImageGlobel::GlobalThresholdResult,127,255,cv::THRESH_BINARY | function)));
       ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::GlobalThresholdResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
       QDateTime currentDateTime = QDateTime::currentDateTime();
       QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
       ui->textEdit_Infor->append(QString("图像全阈值计算结果:%1，图像全阈值计算并绘制成功").arg(formattedTime));
       }
}
//自适应阈值计算
void ImbaProcess::on_checkBox_AdaptiveComputing_stateChanged(int state){
    //选中则计算
    if(state==Qt::Checked){
       //判断卷积核是否是奇数，否则加1再进行计算
       if(ui->spinBox_AdaptiveCoreSize->value()%2==0){
          ui->spinBox_AdaptiveCoreSize->setValue(ui->spinBox_AdaptiveCoreSize->value()+1);
       }
       cv::Mat imageVideo;//存储视频或者本地图片中间变量
       //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
       if(ui->checkBox_BinaryImageVideo->checkState()==Qt::Unchecked){
           imageVideo=cv::imread(ui->FilePatch->text().toStdString());
           if(imageVideo.empty()){
               QDateTime currentDateTime = QDateTime::currentDateTime();
               QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
               ui->textEdit_Infor->append(QString("本地图片数据应用:%1，自适应阈值计算失败，文件路径为空").arg(formattedTime));
               ui->Lab_Image->clear();
               return;
              }
         }else if (ui->checkBox_BinaryImageVideo->checkState()==Qt::Checked){
           //先判断当前视频Label
           if(ui->label_Camera->pixmap()!=nullptr ){
             const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
             imageVideo=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
             }
          }
         //阈值计算方法
         int function;
         if(ui->spinBox_AdaptiveMode->value()==0){
                function=cv::ADAPTIVE_THRESH_MEAN_C;//均值自适应
         }else if(ui->spinBox_AdaptiveMode->value()==1){
                function=cv::ADAPTIVE_THRESH_GAUSSIAN_C;//高斯自适应
         }
         //将图像转换为单通道图像
         cv::cvtColor(imageVideo,imageVideo,cv::COLOR_BGR2GRAY);
         //自适应阈值计算
         cv::adaptiveThreshold(imageVideo,BasicImageGlobel::adaptiveResult,255.0,function,ui->spinBox_AdaptiveFunction->value(),ui->spinBox_AdaptiveCoreSize->value(),ui->doubleSpinBox_AdaptiveConstant->value());
         //阈值计算并将阈值结果写入显示区
         ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::adaptiveResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
         QDateTime currentDateTime = QDateTime::currentDateTime();
         QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
         ui->textEdit_Infor->append(QString("自适应阈值计算结果:%1，自适应阈值计算并绘制成功").arg(formattedTime));
  }

}
//去噪与二值化
void ImbaProcess::on_checkBox_Binaryzation_stateChanged(int state){
    //选中则计算
    if(state==Qt::Checked &&ui->checkBox_BinaryHsv->checkState()==Qt::Unchecked){//不使用HSV
        //首先判定二值化之前取用哪种预处理方式
        cv::Mat PretreatmentResult;
        if (ui->spinBox_BinaryUseResult->value()==0){//卷积模糊结果
            PretreatmentResult=BasicImageGlobel::ConvolutionaAmbiguityResult.clone();
        }else if(ui->spinBox_BinaryUseResult->value()==1){//自定义滤波结果
            PretreatmentResult=BasicImageGlobel::CustomFiltering.clone();
        }else if(ui->spinBox_BinaryUseResult->value()==2){//高斯双边结果
            PretreatmentResult=BasicImageGlobel::BliateraResult.clone();
        }
        if(PretreatmentResult.empty()){
            QDateTime currentDateTime = QDateTime::currentDateTime();
            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
            ui->textEdit_Infor->append(QString("图像二值化结果:%1，图像二值化失败，请先按照要求进行图像预处理").arg(formattedTime));
            ui->Lab_Image->clear();
            return;
        }
        //直接进行图像二值化
        //阈值计算方法
        int function;
        if(ui->spinBox_BinarylaterFunction->value()==0){
               function=cv::ADAPTIVE_THRESH_MEAN_C;//均值自适应
        }else if(ui->spinBox_BinarylaterFunction->value()==1){
               function=cv::ADAPTIVE_THRESH_GAUSSIAN_C;//高斯自适应
        }
        cv::cvtColor(PretreatmentResult,PretreatmentResult,cv::COLOR_BGR2GRAY);
        cv::threshold(PretreatmentResult,BasicImageGlobel::BinaryResult,127,255,cv::THRESH_BINARY|function);
        ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::BinaryResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_Infor->append(QString("图像二值化结果:%1，图像预处理和二值化计算并绘制成功").arg(formattedTime));
   }else if(state==Qt::Checked &&ui->checkBox_BinaryHsv->checkState()==Qt::Checked){//使用HSV
        //将原图转换为HSV
        cv::Mat imageVideo,mask;//存储视频或者本地图片中间变量
        //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
        if(ui->checkBox_BinaryImageVideo->checkState()==Qt::Unchecked){
            imageVideo=cv::imread(ui->FilePatch->text().toStdString());
            if(imageVideo.empty()){
                QDateTime currentDateTime = QDateTime::currentDateTime();
                QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                ui->textEdit_Infor->append(QString("本地图片数据应用:%1，HSV计算失败，文件路径为空").arg(formattedTime));
                ui->Lab_Image->clear();
                return;
               }
          }else if (ui->checkBox_BinaryImageVideo->checkState()==Qt::Checked){
            //先判断当前视频Label
            if(ui->label_Camera->pixmap()!=nullptr ){
              const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
              imageVideo=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
              }
           }
           cv::Mat Hsvimage;
           cv::cvtColor(imageVideo,Hsvimage,cv::COLOR_BGR2HSV);
           int Hmin=ui->spinBox_Hminvalue->value();
           int Smin=ui->spinBox_Sminvalue->value();
           int Vmin=ui->spinBox_Vminvalue->value();
           int Hmax=ui->spinBox_Hmaxvalue->value();
           int Smax=ui->spinBox_Smaxvalue->value();
           int Vmax=ui->spinBox_Vmaxvalue->value();
           //通过imrange取出对应颜色的区域
           cv::inRange(Hsvimage,cv::Scalar(Hmin,Smin,Vmin),cv::Scalar(Hmax,Smax,Vmax),mask);
           cv::bitwise_and(imageVideo,imageVideo,BasicImageGlobel::BinaryResult,mask);
           //阈值计算并将阈值结果写入显示区
           ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::BinaryResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
           QDateTime currentDateTime = QDateTime::currentDateTime();
           QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
           ui->textEdit_Infor->append(QString("去噪与二值化计算结果:%1，HSV二值化计算并绘制成功").arg(formattedTime));
    }
}
//联通组件标记
void ImbaProcess::on_checkBox_ConnectingElement_stateChanged(int state){   
    //选中则计算
    if(state==Qt::Checked){
        //首先判定联通组件扫描之前取用哪种预处理方式
        cv::Mat connectResult;
        if (ui->spinBox_UseBinaryResult->value()==0){//图像阈值化分割结果
            connectResult=BasicImageGlobel::ThresholdResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==1){//全阈值化分割结果
            connectResult=BasicImageGlobel::GlobalThresholdResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==2){//自适应阈值计算结果
            connectResult=BasicImageGlobel::adaptiveResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==3){//去噪与二值化结果
            connectResult=BasicImageGlobel::BinaryResult.clone();
        }
        if(connectResult.empty()){
            QDateTime currentDateTime = QDateTime::currentDateTime();
            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
            ui->textEdit_Infor->append(QString("图像联通组件扫描结果:%1，联通组件扫描失败，请先按照要求进行图像预处理得到二值图像").arg(formattedTime));
            ui->Lab_Image->clear();
            return;
            }
         //联通组件扫描
        cv::Mat labelimage,state,centteroids;
        int num=cv::connectedComponentsWithStats(connectResult,labelimage,state,centteroids,8);
        cv::cvtColor(connectResult,connectResult,cv::COLOR_GRAY2BGR);
        int Connectnum=0;//有效联通组件数量
        for (int i=0;i< num; i++) {
            int CenterPointx=centteroids.at<double>(i,0);//质心X坐标
            int CenterPointy=centteroids.at<double>(i,1);//质心Y坐标
            int left=state.at<int>(i,cv::CC_STAT_LEFT);//左上角x坐标值
            int top=state.at<int>(i,cv::CC_STAT_TOP);//左上角y坐标值
            int width=state.at<int>(i,cv::CC_STAT_WIDTH);//最小外接矩形宽
            int height=state.at<int>(i,cv::CC_STAT_HEIGHT);//最小外接矩形高
            int area  =state.at<int>(i,cv::CC_STAT_AREA);//联通组件面积大小
            cv::Rect rect(left,top,width,height);//外接矩形
            if(ui->checkBox_AreaFilter->checkState()==Qt::Checked && area>ui->spinBox_AreaDownlimit->value()&&area<ui->spinBox_AreaUplimit->value()){
                Connectnum++;
                cv::putText(connectResult,QString("%1").arg(Connectnum).toStdString(),cv::Point(CenterPointx,CenterPointy),cv::FONT_HERSHEY_SIMPLEX,0.9,cv::Scalar(255,0,0),2,8);
                cv::rectangle(connectResult,rect,cv::Scalar(0,255,0),2,8);//绘制外接矩形
                cv::circle(connectResult,cv::Point(CenterPointx,CenterPointy),2,cv::Scalar(0,0,255),2,8);//绘制中心圆
                ui->textEdit_Infor->append(QString("组件:%1的面积是：%2").arg(Connectnum).arg(area));//向信息区输出每个组件的面
                ui->lineEdit_ConnnectNumber->setText(QString().setNum(Connectnum));
            }else if(ui->checkBox_AreaFilter->checkState()==Qt::Unchecked){
                cv::putText(connectResult,QString("%1").arg(i).toStdString(),cv::Point(CenterPointx,CenterPointy),cv::FONT_HERSHEY_SIMPLEX,0.8,cv::Scalar(255,0,0),2,8);
                cv::rectangle(connectResult,rect,cv::Scalar(0,255,0),2,8);//绘制外接矩形
                cv::circle(connectResult,cv::Point(CenterPointx,CenterPointy),2,cv::Scalar(0,0,255),2,8);//绘制中心圆
                ui->textEdit_Infor->append(QString("组件:%1的面积是：%2").arg(i).arg(area));//向信息区输出每个组件的面
                ui->lineEdit_ConnnectNumber->setText(QString().setNum(i));
            }

           }
        BasicImageGlobel::connectComResult=connectResult;
        //联通组件结果写入显示区
        ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::connectComResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_Infor->append(QString("联通组件扫描计算结果:%1，联通组件扫描计算并绘制成功").arg(formattedTime));
     }
}
//轮廓发现绘制与测量
void ImbaProcess::on_checkBox_ContourDiscovery_stateChanged(int state){
    //选中则计算
    if(state==Qt::Checked){
        //首先判定联通组件扫描之前取用哪种预处理方式
        cv::Mat ContourResult;
        if (ui->spinBox_UseBinaryResult->value()==0){//图像阈值化分割结果
            ContourResult=BasicImageGlobel::ThresholdResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==1){//全阈值化分割结果
            ContourResult=BasicImageGlobel::GlobalThresholdResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==2){//自适应阈值计算结果
            ContourResult=BasicImageGlobel::adaptiveResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==3){//去噪与二值化结果
            ContourResult=BasicImageGlobel::BinaryResult.clone();
        }
        if(ContourResult.empty()){
            QDateTime currentDateTime = QDateTime::currentDateTime();
            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
            ui->textEdit_Infor->append(QString("轮廓发现绘制与测量结果:%1，轮廓发现失败，请先按照要求进行图像预处理得到二值图像").arg(formattedTime));
            ui->Lab_Image->clear();
            return;
            }
           //发现轮廓
           std::vector<std::vector<cv::Point>> Contours;//轮廓点集集合
           std::vector<cv::Vec4i> Hierarchy;//轮廓层次信息
           cv::findContours(ContourResult,Contours,Hierarchy,ui->spinBox_ContoursMode->value(),ui->spinBox_ContoursFunction->value());//发现轮廓的函数
           cv::cvtColor(ContourResult,ContourResult,cv::COLOR_GRAY2BGR);//转换为彩色图像
           cv::drawContours(ContourResult,Contours,-1,cv::Scalar(0,255,0),2,8);//绘制轮廓
           BasicImageGlobel::ContoursResult=Contours;//轮廓点集集合映射出去别的地方使用
           //轮廓测量
           if(ui->checkBox_ContoursMeasure->checkState()==Qt::Checked){
               for (int i=0;i<Contours.size();i++ ){
                    cv::Rect rect=cv::boundingRect(Contours[i]);//计算每个轮廓的最小外接矩形
                    double area=cv::contourArea(Contours[i]);//计算轮廓点集面积
                    double Lenth=cv::arcLength(Contours[i],true);//计算闭合区域周长
                    cv::putText(ContourResult,QString("Area:%1").arg(area).toStdString(),cv::Point(rect.x,rect.y),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,0,255),1,8);
                    cv::putText(ContourResult,QString("Lenth:%1").arg(Lenth).toStdString(),cv::Point(rect.x,rect.y+15),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,0,0),1,8);
                    ui->textEdit_Infor->append(QString("轮廓:%1的面积是：%2").arg(i).arg(area));//向信息区输出每个组件的面
                    ui->textEdit_Infor->append(QString("轮廓:%1的周长是：%2").arg(i).arg(Lenth));//向信息区输出每个组件的面
               }
           }
           BasicImageGlobel::FindDrwContours=ContourResult;
           //轮廓发现绘制与测量结果写入显示区
           ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::FindDrwContours).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
           QDateTime currentDateTime = QDateTime::currentDateTime();
           QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
           ui->textEdit_Infor->append(QString("轮廓发现绘制与测量计算结果:%1，轮廓发现绘制与测量计算并绘制成功").arg(formattedTime));

     }
}
//拟合与逼近
void ImbaProcess::on_checkBox_FittingApproximation_stateChanged(int state){
    //选中则计算
    if(state==Qt::Checked){
         //首先将Canny边缘处理结果或者二值化结果映射过来
          cv::Mat FitGianOnResult;
         //首先判定轮廓逼近之前取用哪种预处理方式
         if (ui->checkBox_approxpolyDP->checkState()==Qt::Checked){
             if (ui->spinBox_UseBinaryResult->value()==0){//图像阈值化分割结果
                 FitGianOnResult=BasicImageGlobel::ThresholdResult.clone();
             }else if(ui->spinBox_UseBinaryResult->value()==1){//全阈值化分割结果
                 FitGianOnResult=BasicImageGlobel::GlobalThresholdResult.clone();
             }else if(ui->spinBox_UseBinaryResult->value()==2){//自适应阈值计算结果
                 FitGianOnResult=BasicImageGlobel::adaptiveResult.clone();
             }else if(ui->spinBox_UseBinaryResult->value()==3){//去噪与二值化结果
                 FitGianOnResult=BasicImageGlobel::BinaryResult.clone();
             }
          }else if(ui->checkBox_approxpolyDP->checkState()==Qt::Unchecked){
                  FitGianOnResult=BasicImageGlobel::CannyResult.clone();
         }
        if(FitGianOnResult.empty()){
            QDateTime currentDateTime = QDateTime::currentDateTime();
            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
            ui->textEdit_Infor->append(QString("拟合与逼近结果:%1，拟合与逼近失败，请先按照要求进行Canny边缘处理").arg(formattedTime));
            ui->Lab_Image->clear();
            return;
            }
           //轮廓发现
           cv::Mat k=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));//获取结构性元素
           cv::dilate(FitGianOnResult,FitGianOnResult,k);//最大值滤波(也就是膨胀操作)，相当于把边缘连接起来
           std::vector<std::vector<cv::Point>>Contours;//轮廓点集集合
           std::vector<cv::Vec4i>heirchy;//轮廓层次信息
           cv::findContours(FitGianOnResult,Contours,heirchy,ui->spinBox_FittingCtoursMode->value(),ui->spinBox_FitingContoursFunction->value());//发现轮廓
           cv::cvtColor(FitGianOnResult,FitGianOnResult,cv::COLOR_GRAY2BGR);
        //判断是哪种方式进行拟合与逼近
       if(ui->checkBox_fitEllipse->checkState()==Qt::Checked){//拟合圆
           for (int i=0;i<Contours.size();i++) {
                if(Contours[i].size()<ui->spinBox_ContoursFilting->value()){//轮廓点集滤波
                    continue;
                }
                cv::RotatedRect rrt=cv::fitEllipse(Contours[i]);
                cv::Point center=rrt.center;
                float width =rrt.size.width;
                float height =rrt.size.height;
                cv::ellipse(FitGianOnResult,rrt,cv::Scalar(0,0,255),2,8);
           }
       }else if(ui->checkBox_fitLine->checkState()==Qt::Checked){//拟合直线
           //直线拟合,首先找到这个直线上中心点或者任意与真实数据点拟合之后最匹配的点的坐标和斜率 任意直线上的点遵循公式 y = kx +b
           cv::Vec4f line1;
           for (int i=0;i<Contours.size();i++ ) {
               cv::fitLine(Contours[i],line1,cv::DIST_L1,0,0.01,0.01);
               float vx =line1[0];//向量x
               float vy =line1[1];//向量y 由公式y=kx 可得该直线的斜率
               float x0 =line1[2];//x坐标值
               float y0 =line1[3];//y坐标值
               //计算直线斜率和截距 y=mx+b 其中斜率为 m为斜率 b是当x=0时y的截距 x的截距为 - b/m
               float m=vy/vx;   //斜率m
               float b= y0-m*x0;//截距b
               //通过循环找到点集中的最大点和最小点,然后通过函数将其绘制出来
               int minx =0;   int miny =10000;//任意最大初始值
               int mmaxx =0;  int maxy  =0;
               for(int j=0;j<Contours[i].size();j++){
                 cv::Point midPoint=Contours[i][j];
                 //判断y轴最小点
                 if (miny>midPoint.y){
                     miny=midPoint.y;
                 }
                 //判断y轴最大点
                 if (maxy<midPoint.y){
                     maxy=midPoint.y;
                 }
               }
               //根据公式 y =mx +b 上面计算出了y轴最大的值,也有了斜率，那么也就可以将x轴计算出来
               minx=(miny-b)/m;
               mmaxx=(maxy-b)/m;
               //绘制线段(有了斜率，截距，x值，y值那么就可以形成一条直线)
               line(FitGianOnResult,cv::Point(minx,miny),cv::Point(mmaxx,maxy),cv::Scalar(0,0,255),2,cv::LINE_AA);
           }

       }else if(ui->checkBox_approxpolyDP->checkState()==Qt::Checked){//轮廓逼近
           for (int i=0;i<Contours.size();i++ ) {
               std::vector<cv::Point>Pts;//逼近点集
               cv::approxPolyDP(Contours[i],Pts,ui->spinBox_approxpolyDPMaxDis->value(),ui->radioButton_NoOrisRegion->isChecked());//开始逼近
               for (int j=0;j<Pts.size();j++) {
                   cv::circle(FitGianOnResult,Pts[j],2,cv::Scalar(0,0,255),2,8);
               }
           }
       }
       //拟合与逼近结果写入显示区
       BasicImageGlobel::FittingApproximationResults=FitGianOnResult.clone();
       ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::FittingApproximationResults).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
       QDateTime currentDateTime = QDateTime::currentDateTime();
       QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
       ui->textEdit_Infor->append(QString("拟合与逼近计算结果:%1，拟合与逼近计算并绘制成功").arg(formattedTime));
    }
}
//轮廓分析
void ImbaProcess::on_checkBox_ContourAnalyse_stateChanged(int state){
    //首先将Canny边缘处理结果或者二值化结果映射过来
     cv::Mat Contoursanalyse;
    //首先判定轮廓逼近之前取用哪种预处理方式
    if (ui->checkBox_CannyORBinary->checkState()==Qt::Checked){
        if (ui->spinBox_UseBinaryResult->value()==0){//图像阈值化分割结果
            Contoursanalyse=BasicImageGlobel::ThresholdResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==1){//全阈值化分割结果
            Contoursanalyse=BasicImageGlobel::GlobalThresholdResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==2){//自适应阈值计算结果
            Contoursanalyse=BasicImageGlobel::adaptiveResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==3){//去噪与二值化结果
            Contoursanalyse=BasicImageGlobel::BinaryResult.clone();
        }
     }else if(ui->checkBox_CannyORBinary->checkState()==Qt::Unchecked){
             Contoursanalyse=BasicImageGlobel::CannyResult.clone();
    }
    if(Contoursanalyse.empty()){
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_Infor->append(QString("轮廓分析结果:%1，轮廓分析失败，请先按照要求进行Canny边缘处理或者图像二值化操作").arg(formattedTime));
        ui->Lab_Image->clear();
        return;
        }
    //选中则计算
    if (state==Qt::Checked){

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::Mat k=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));//获取结构性元素
        cv::dilate(Contoursanalyse,Contoursanalyse,k);//最大值滤波(也就是膨胀操作)，相当于把边缘连接起来
        findContours(Contoursanalyse,contours,hierarchy,ui->spinBox_FittingCtoursMode->value(),ui->spinBox_FitingContoursFunction->value());
        cvtColor(Contoursanalyse,Contoursanalyse,cv::COLOR_GRAY2BGR);
        //循环计算
        for(int i=0;i<contours.size();i++){
            if (contours[i].size()<50){
                continue;
                }
          cv::Rect rect =boundingRect(contours[i]);//最大外接矩形
          cv::RotatedRect rrt =minAreaRect(contours[i]);//最小外接矩形
          cv::rectangle(Contoursanalyse,rect,cv::Scalar(0,0,255),2,8);//绘制最大外界矩形
          cv::rectangle(Contoursanalyse,cv::Rect(rrt.center.x-(rrt.size.width/2),rrt.center.y-(rrt.size.height/2),rrt.size.width,rrt.size.height),cv::Scalar(0,255,0),2,8);//绘制最小外界矩形(这里计算其实是忽略了矩形的角度的)
          std::vector<cv::Point>Hull;//计算凸包
          convexHull(contours[i],Hull);
          double hull_area = contourArea(Hull);//凸包轮廓面积
          double box_area  = rect.height*rect.width;//外接矩形的面积
          double area =contourArea(contours[i]);//点集轮廓面积
          double aspect_radio =static_cast<double> (rrt.size.width) /static_cast<double> (rrt.size.height);//计算纵横比
          double ext = area/box_area;//计算延展度
          double solid = area/hull_area;//计算实密度
          //生成掩膜并计算像素均值
          cv::Mat mask = cv::Mat::zeros(Contoursanalyse.size(),CV_8UC1);
          mask.setTo(cv::Scalar(0));
          drawContours(mask,contours,-1,cv::Scalar(255),cv::FILLED,cv::LINE_AA);
          cv::Scalar meanValve = mean(Contoursanalyse,mask);
          //文本显示
          //putText(Contoursanalyse,cv::format("aspect_radio:%.4f",aspect_radio),cv::Point(rect.x,rect.y),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);
          //putText(Contoursanalyse,cv::format("ext:%.4f"         ,ext),cv::Point(rect.x,rect.y+20),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);
          //putText(Contoursanalyse,cv::format("solid:%.4f"       ,solid),cv::Point(rect.x,rect.y+40),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);
          //putText(Contoursanalyse,cv::format("meanValve:%d, %d ,%d",(int)meanValve[0],(int)meanValve[1],(int)meanValve[2]),cv::Point(rect.x,rect.y+60),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);
          ui->textEdit_Infor->append(QString("轮廓%1纵横比:%2，").arg(i).arg(aspect_radio));//纵横比输出
          ui->textEdit_Infor->append(QString("轮廓%1延展度:%2，").arg(i).arg(ext));//延展度
          ui->textEdit_Infor->append(QString("轮廓%1实密度:%2，").arg(i).arg(solid));//实密度
          ui->textEdit_Infor->append(QString("轮廓%1像素均值:%2,%3,%4").arg(i).arg((int)meanValve[0]).arg((int)meanValve[1]).arg((int)meanValve[2]));//像素均值
        }
         //轮廓分析结果写入显示区
         BasicImageGlobel::ContoursanalyseResult=Contoursanalyse.clone();
         ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::ContoursanalyseResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
         QDateTime currentDateTime = QDateTime::currentDateTime();
         QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
         ui->textEdit_Infor->append(QString("轮廓分析计算结果:%1，轮廓分析计算并绘制成功").arg(formattedTime));
       }
}
//直线检测
void ImbaProcess::on_checkBox_LineDetection_stateChanged(int state){
    //首先将Canny边缘处理结果或者二值化结果映射过来
     cv::Mat LineDetection;
    //首先判定直线检测之前取用哪种预处理方式
    if (ui->checkBox_LineDeCannyORBinary->checkState()==Qt::Checked){
        if (ui->spinBox_UseBinaryResult->value()==0){//图像阈值化分割结果
            LineDetection=BasicImageGlobel::ThresholdResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==1){//全阈值化分割结果
            LineDetection=BasicImageGlobel::GlobalThresholdResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==2){//自适应阈值计算结果
            LineDetection=BasicImageGlobel::adaptiveResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==3){//去噪与二值化结果
            LineDetection=BasicImageGlobel::BinaryResult.clone();
        }
     }else if(ui->checkBox_LineDeCannyORBinary->checkState()==Qt::Unchecked){
             LineDetection=BasicImageGlobel::CannyResult.clone();
    }
    if(LineDetection.empty()){
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_Infor->append(QString("直线检测结果:%1，直线检测失败，请先按照要求进行Canny边缘处理或者图像二值化操作").arg(formattedTime));
        ui->Lab_Image->clear();
        return;
        }
    //选中则计算
    if (state==Qt::Checked){
        std::vector<cv::Vec4i>LineP;
        //概率霍夫直线检测
        cv::HoughLinesP(LineDetection,LineP,1,CV_PI/180,ui->spinBox_DetectionResultThreshold->value(),ui->spinBox_MinLineLenth->value(),ui->spinBox_LineMaxInterval->value());
        cv::cvtColor(LineDetection,LineDetection,cv::COLOR_GRAY2BGR);
        for(int i=0;i<LineP.size();i++){
           cv::Point p1=cv::Point(LineP[i][0],LineP[i][1]);
           cv::Point p2=cv::Point(LineP[i][2],LineP[i][3]);
           cv::line(LineDetection,p1,p2,cv::Scalar(0,0,255),2,8);
        }
        //霍夫直线检测结果写入显示区
        BasicImageGlobel::LineDetectionResult=LineDetection.clone();
        ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::LineDetectionResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_Infor->append(QString("轮廓分析计算结果:%1，轮廓分析计算并绘制成功").arg(formattedTime));
       }
}
//霍夫圆检测
void ImbaProcess::on_checkBox_HoffCircleDetection_stateChanged(int state){
    //首先将二值化结果映射过来
    cv::Mat CircleDetection=BasicImageGlobel::ConvolutionaAmbiguityResult.clone();
    if(CircleDetection.empty()){
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_Infor->append(QString("霍夫圆检测结果:%1，霍夫圆检测失败，请先按照要求进行图像卷积滤波操作").arg(formattedTime));
        ui->Lab_Image->clear();
        return;
        }
    //选中则计算
    if(state==Qt::Checked){
      //霍夫圆检测
       cv::cvtColor(CircleDetection,CircleDetection,cv::COLOR_BGR2GRAY);
       std::vector<cv::Vec3f>Circle;
       cv::HoughCircles(CircleDetection,Circle,cv::HOUGH_GRADIENT,ui->spinBox_resolutionRatio->value(),ui->spinBox_MinSpeaceBettween->value(),ui->spinBox_CircleHeightThreshold->value(),ui->spinBox_CircleLowThreshold->value(),ui->spinBox_MinRadius->value(),ui->spinBox_MaxRadius->value());
       cv::cvtColor(CircleDetection,CircleDetection,cv::COLOR_GRAY2BGR);
       for(int i=0;i<Circle.size();i++){
          cv::Point Center=cv::Point(Circle[i][0],Circle[i][1]);//圆心
          int Radius=Circle[i][2];
          cv::circle(CircleDetection,Center,Radius,cv::Scalar(0,0,255),2,8,0);
       }
      //霍夫圆检测检测结果写入显示区
      BasicImageGlobel::CircleDetectionResult=CircleDetection.clone();
      ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::CircleDetectionResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
      QDateTime currentDateTime = QDateTime::currentDateTime();
      QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
      ui->textEdit_Infor->append(QString("霍夫圆检测结果:%1，霍夫圆检测并绘制成功").arg(formattedTime));
    }

}

//最大内接圆和最小外接圆
void ImbaProcess::on_checkBox_BigSmallCircle_stateChanged(int state){
    //首先将二值化结果映射过来
     cv::Mat MinMaxCircle;
    //首先判定最大内接圆和最小外接圆之前取用哪种预处理方式
    if (state==Qt::Checked){
        if (ui->spinBox_UseBinaryResult->value()==0){//图像阈值化分割结果
            MinMaxCircle=BasicImageGlobel::ThresholdResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==1){//全阈值化分割结果
            MinMaxCircle=BasicImageGlobel::GlobalThresholdResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==2){//自适应阈值计算结果
            MinMaxCircle=BasicImageGlobel::adaptiveResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==3){//去噪与二值化结果
            MinMaxCircle=BasicImageGlobel::BinaryResult.clone();
        }
        if(MinMaxCircle.empty()){
            QDateTime currentDateTime = QDateTime::currentDateTime();
            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
            ui->textEdit_Infor->append(QString("最大内接圆和最小外接圆结果:%1，最大内接圆和最小外接圆检测失败，请先按照要求进行图像二值化操作").arg(formattedTime));
            ui->Lab_Image->clear();
            return;
            }
    }

    //选中则计算
    if (state==Qt::Checked){
        //首先还是需要找轮廓
        std::vector<std::vector<cv::Point>> Contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::Mat k=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));//获取结构性元素
        cv::dilate(MinMaxCircle,MinMaxCircle,k);//最大值滤波(也就是膨胀操作)，相当于把边缘连接起来
        cv::findContours(MinMaxCircle,Contours,hierarchy,ui->spinBox_MinMaxContoursMode->value(),ui->spinBox_MinMaxContoursFunction->value());
        cvtColor(MinMaxCircle,MinMaxCircle,cv::COLOR_GRAY2BGR);
        for (int t=0;t<Contours.size();t++ ) {
            if(ui->checkBox_MaxMinCircle->checkState()==Qt::Unchecked){//绘制最小圆
                cv::Point2f center;
                float radius;
                cv::minEnclosingCircle(Contours[t],center,radius);//寻找最小外接圆
                cv::circle(MinMaxCircle,center,radius,cv::Scalar(0,0,255),2,8,0);
            }else if(ui->checkBox_MaxMinCircle->checkState()==Qt::Checked){//绘制最大圆
                //点多边形测试
                cv::Mat Testimage(MinMaxCircle.size(),CV_32F);
                for (int i=0;i<MinMaxCircle.rows;i++) {
                    for (int j=0;j<MinMaxCircle.cols;j++ ) {
                         Testimage.at<float>(i,j)=(float)cv::pointPolygonTest(Contours[t],cv::Point2f((float)j,(float)i),true);//遍历计算图像中的每个点距离轮廓的位置
                    }
                }
                //绘制最大内接圆
                double MinValue,MaxValue;
                cv::Point MaxValuePos;
                cv::minMaxLoc(Testimage,&MinValue,&MaxValue,NULL,&MaxValuePos);
                MaxValue=cv::abs(MaxValue);
                MinValue=cv::abs(MinValue);
                cv::circle(MinMaxCircle,MaxValuePos,MaxValue,cv::Scalar(255,0,0),2,8,0);
            }
        }
        //最大内接圆和最小外接圆检测结果写入显示区
        BasicImageGlobel::CircleDetectionResult=MinMaxCircle.clone();
        ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::CircleDetectionResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_Infor->append(QString("最大内接圆和最小外接圆检测结果:%1，最大内接圆和最小外接圆检测并绘制成功").arg(formattedTime));
      }


}
//轮廓匹配
void ImbaProcess::on_BtnContoursMatchShapopen_clicked()
{
    //待查找对象路径
    QString lastPath = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    QString fileName = QFileDialog::getOpenFileName(this, "请选择图片", lastPath, "图片(*.png *.jpg);;");
    if(fileName.isEmpty())
    {
        //加载失败，向信息区中写入信息
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_Infor->append(QString("本地图片数据采集:%1，图像加载失败，文件路径为空").arg(formattedTime));
        ui->Lab_Image->clear();
        ui->FilePatch->clear();
        return;
    }
    //将路径加载至文件路径显示栏中
    ui->FilePatch_ContoursMatchShap->setText(fileName);
}

void ImbaProcess::on_checkBox_ContourMaching_stateChanged(int state){
    //目标查找区图像,查找图像转换为灰度图像
     cv::Mat Targetimgae,Findimage;
     Findimage=cv::imread(ui->FilePatch->text().toStdString());
     Targetimgae=cv::imread(ui->FilePatch_ContoursMatchShap->text().toStdString());
     //图像取反并转换通道
     cv::bitwise_not(Targetimgae,Targetimgae);
     cv::bitwise_not(Findimage,Findimage);
     cv::cvtColor(Targetimgae,Targetimgae,cv::COLOR_BGR2GRAY);
     cv::cvtColor(Findimage,Findimage,cv::COLOR_BGR2GRAY);

    //首先判定轮廓匹配之前取用哪种预处理方式
    if (state==Qt::Checked){
        if(Targetimgae.empty()||Findimage.empty()){//目标图像和待查找对象不能为空
            QDateTime currentDateTime = QDateTime::currentDateTime();
            QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
            ui->textEdit_Infor->append(QString("轮廓匹配结果:%1，轮廓匹配失败，请先按照要求进行图像二值化且目标对象不能为空操作").arg(formattedTime));
            ui->Lab_Image->clear();
            return;
        }
        //轮廓发现
        std::vector<std::vector<cv::Point>>ContoursTarget,ContoursFind;
        std::vector<cv::Vec4i>HierarchyTarget,HierarchyFind;
        cv::findContours(Targetimgae,ContoursTarget,HierarchyTarget,ui->spinBox_MachContoursMode->value(),ui->spinBox_MachContoursFunction->value());
        cv::findContours(Findimage,ContoursFind,HierarchyFind,ui->spinBox_MachContoursMode->value(),ui->spinBox_MachContoursFunction->value());
        cv::cvtColor(Targetimgae,Targetimgae,cv::COLOR_GRAY2BGR);
        cv::cvtColor(Findimage,Findimage,cv::COLOR_GRAY2BGR);

        for (int i=0;i<ContoursTarget.size();i++ ) {//目标轮廓
            for (int j=0;j<ContoursFind.size();j++ ) {//待查找区域轮廓
                //计算几何矩
                cv::Moments TargetMoment=cv::moments(ContoursTarget[i]);//目标图像几何矩
                cv::Moments FindMoment  =cv::moments(ContoursFind[j]);//查找图像集合矩
                //计算胡矩
                cv::Mat TargetHu,FindHu;
                cv::HuMoments(TargetMoment,TargetHu);
                cv::HuMoments(FindMoment,FindHu);
                //轮廓匹配(返回匹配距离)
                double dist=cv::matchShapes(TargetHu,FindHu,cv::CONTOURS_MATCH_I1,0);
                //小于设定匹配阈值，那么就判定为找到匹配对象了
                if(dist<ui->DoublespinBox_MachshapeThreshold->value()){
                   //将找到的轮廓绘制出来
                   cv::Rect Resultrect=cv::boundingRect(ContoursFind[j]);
                   cv::rectangle(Findimage,Resultrect,cv::Scalar(0,0,255),2,8,0);
                }
            }
        }
        //轮廓匹配结果写入显示区
        BasicImageGlobel::ContoursMatchShapResult=Findimage.clone();
        ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::ContoursMatchShapResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_Infor->append(QString("轮廓匹配结果:%1，轮廓匹配并绘制成功").arg(formattedTime));
    }

}
//最大轮廓与关键点编码
void ImbaProcess::on_checkBox_KeyPiontCoding_stateChanged(int state){
    //首先将二值化结果映射过来
     cv::Mat MaxContours;
    //首先判定最大轮廓与关键点编码之前取用哪种预处理方式
    if (state==Qt::Checked){
        if (ui->spinBox_UseBinaryResult->value()==0){//图像阈值化分割结果
            MaxContours=BasicImageGlobel::ThresholdResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==1){//全阈值化分割结果
            MaxContours=BasicImageGlobel::GlobalThresholdResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==2){//自适应阈值计算结果
            MaxContours=BasicImageGlobel::adaptiveResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==3){//去噪与二值化结果
            MaxContours=BasicImageGlobel::BinaryResult.clone();
        }
    if(MaxContours.empty()){
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_Infor->append(QString("关键点编码检测结果:%1，关键点编码检测失败，请先按照要求进行图像二值化操作").arg(formattedTime));
        ui->Lab_Image->clear();
        return;
        }
       //轮廓发现
       std::vector<std::vector<cv::Point>>Contours;
       std::vector<cv::Vec4i>Hierarchy;
       cv::findContours(MaxContours,Contours,Hierarchy,ui->spinBox_MachContoursMode->value(),ui->spinBox_MachContoursFunction->value());//轮廓发现
       //寻找最大轮廓
       double MaxArea=0.0;
       int index;
       for(int i=0;i<Contours.size();i++){
          double area=cv::contourArea(Contours[i]);
          if(area>MaxArea){
              MaxArea=area;//最大面积
              index=i;//最大面积的索引
          }
       }
       ui->lineEdit_Maxarea->setText(QString().setNum(MaxArea));//将最大轮廓面积写入文本显示区
       //绘制轮廓
       cv::Mat Result=cv::Mat::zeros(MaxContours.size(),CV_8UC3);
       cv::drawContours(Result,Contours,index,cv::Scalar(0,0,255),2,8);
       //关键点编码
       std::vector<cv::Point>pts;
       cv::approxPolyDP(Contours[index],pts,ui->spinBoxMaxContoursDis->value(),ui->radioButton_isornoCloseArea->isChecked());
       for(int j=0;j<pts.size();j++){
           cv::circle(Result,pts[j],2,cv::Scalar(255,0,0),2,8);
       }
       //最大轮廓与关键点编码结果写入显示区
       BasicImageGlobel::MaxContourKeyPointCodResult=Result.clone();
       ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::MaxContourKeyPointCodResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
       QDateTime currentDateTime = QDateTime::currentDateTime();
       QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
       ui->textEdit_Infor->append(QString("最大轮廓与关键点编码结果:%1，最大轮廓与关键点编码绘制成功").arg(formattedTime));
    }
}

//凸包检测
void ImbaProcess::on_checkBox_ConvexDetection_stateChanged(int state){
    //首先将二值化结果映射过来
     cv::Mat HullDection;
    //首先判定最大轮廓与关键点编码之前取用哪种预处理方式
    if (state==Qt::Checked){
        if (ui->spinBox_UseBinaryResult->value()==0){//图像阈值化分割结果
            HullDection=BasicImageGlobel::ThresholdResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==1){//全阈值化分割结果
            HullDection=BasicImageGlobel::GlobalThresholdResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==2){//自适应阈值计算结果
            HullDection=BasicImageGlobel::adaptiveResult.clone();
        }else if(ui->spinBox_UseBinaryResult->value()==3){//去噪与二值化结果
            HullDection=BasicImageGlobel::BinaryResult.clone();
        }
    if(HullDection.empty()){
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_Infor->append(QString("关键点编码检测结果:%1，关键点编码检测失败，请先按照要求进行图像二值化操作").arg(formattedTime));
        ui->Lab_Image->clear();
        return;
        }

    //轮廓发现
    std::vector<std::vector<cv::Point>>Contours;
    std::vector<cv::Vec4i>Hierarchy;
    cv::findContours(HullDection,Contours,Hierarchy,ui->spinBox_MachContoursMode->value(),ui->spinBox_MachContoursFunction->value());//轮廓发现
    cv::cvtColor(HullDection,HullDection,cv::COLOR_GRAY2BGR);
    for (int i=0;i<Contours.size();i++ ) {
        //凸包检测函数
        std::vector<cv::Point>Hull;
        cv::convexHull(Contours[i],Hull);
        for (int j=0;j<Hull.size();j++ ) {
             cv::circle(HullDection,Hull[j],2,cv::Scalar(0,0,255),2,8);
         }
     }
    //凸包检测结果写入显示区
    BasicImageGlobel::HullDectionResult=HullDection.clone();
    ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::HullDectionResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
    QDateTime currentDateTime = QDateTime::currentDateTime();
    QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
    ui->textEdit_Infor->append(QString("凸包检测结果:%1，凸包检测绘制成功").arg(formattedTime));
    }
}

                                                                //形态学分析//
/***********************************************************************************************************************************************
*@brief:膨胀与腐蚀操作，开闭操作，形态学梯度，顶帽与黑帽操作，击中与击不中操作，自定义结构性元素操作，距离变换操作，分水岭分割操作
*@date: 2024.07.30
*@param:
***********************************************************************************************************************************************/
//膨胀与腐蚀操作
void ImbaProcess::on_checkBox_StardiateErode_stateChanged(int state){

    if(state==Qt::Checked){//是否进行膨胀与腐蚀操作
        //判断卷积核是否是奇数，否则加1再进行计算
        if(ui->spinBox_MorphCoreSize->value()%2==0){
           ui->spinBox_MorphCoreSize->setValue(ui->spinBox_MorphCoreSize->value()+1);
        }
         cv::Mat image;
            if(ui->checkBox_UseBinaryResult->checkState()==Qt::Checked){
                if (ui->spinBox_dilateErodeBinaryResult->value()==0){//图像阈值化分割结果
                    image=BasicImageGlobel::ThresholdResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==1){//全阈值化分割结果
                    image=BasicImageGlobel::GlobalThresholdResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==2){//自适应阈值计算结果
                    image=BasicImageGlobel::adaptiveResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==3){//去噪与二值化结果
                    image=BasicImageGlobel::BinaryResult.clone();
                }
            }else if(ui->checkBox_UseBinaryResult->checkState()==Qt::Unchecked){
                //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
                if(ui->checkBox_morphoImageVideo->checkState()==Qt::Unchecked){
                  image=cv::imread(ui->FilePatch->text().toStdString());
                  if(image.empty()){
                      QDateTime currentDateTime = QDateTime::currentDateTime();
                      QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                      ui->textEdit_Infor->append(QString("膨胀与腐蚀操作应用:%1，膨胀与腐蚀操作计算失败，文件路径为空").arg(formattedTime));
                      ui->Lab_Image->clear();
                     return;
                  }
               }else if(ui->checkBox_morphoImageVideo->checkState()==Qt::Checked){
                    //先判断当前视频Label
                    if(ui->label_Camera->pixmap()!=nullptr ){
                      const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
                      image=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
                     }
                 }
            }
            //判断是否为空
            if(image.empty()){
                QDateTime currentDateTime = QDateTime::currentDateTime();
                QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                ui->textEdit_Infor->append(QString("膨胀与腐蚀操作检测结果:%1，膨胀与腐蚀操作失败，请先按照要求进行图像预处理操作").arg(formattedTime));
                ui->Lab_Image->clear();
                return;
                }
         //获取结构性元素
         cv::Mat Kernal=cv::getStructuringElement(ui->spinBox_MorphShape->value(),cv::Size(ui->spinBox_MorphCoreSize->value(),ui->spinBox_MorphCoreSize->value()));
         if(ui->checkBox_Morphdilate->checkState()==Qt::Checked){
             //膨胀操作
             cv::dilate(image,BasicImageGlobel::DilateErodeResult,Kernal,cv::Point(-1,-1),ui->spinBox_MorphNum->value());
         }else if(ui->checkBox_checkBox_Erode->checkState()==Qt::Checked){
            //膨胀操作
           cv::erode(image,BasicImageGlobel::DilateErodeResult,Kernal,cv::Point(-1,-1),ui->spinBox_MorphNum->value());
         }else{
             return;
         }
         //膨胀与腐蚀操作写入显示区
         ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::DilateErodeResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
         QDateTime currentDateTime = QDateTime::currentDateTime();
         QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
         ui->textEdit_Infor->append(QString("膨胀与腐蚀结果:%1，膨胀与腐蚀成功").arg(formattedTime));
     }
}

//开闭操作
void ImbaProcess::on_checkBox_StarOpenOrClose_stateChanged(int state){

    if(state==Qt::Checked){//是否进行开闭操作
        //判断卷积核是否是奇数，否则加1再进行计算
        if(ui->spinBox_MorphCoreSize->value()%2==0){
           ui->spinBox_MorphCoreSize->setValue(ui->spinBox_MorphCoreSize->value()+1);
        }
         cv::Mat image;
            if(ui->checkBox_UseBinaryResult->checkState()==Qt::Checked){
                if (ui->spinBox_dilateErodeBinaryResult->value()==0){//图像阈值化分割结果
                    image=BasicImageGlobel::ThresholdResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==1){//全阈值化分割结果
                    image=BasicImageGlobel::GlobalThresholdResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==2){//自适应阈值计算结果
                    image=BasicImageGlobel::adaptiveResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==3){//去噪与二值化结果
                    image=BasicImageGlobel::BinaryResult.clone();
                }
            }else if(ui->checkBox_UseBinaryResult->checkState()==Qt::Unchecked){
                //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
                if(ui->checkBox_morphoImageVideo->checkState()==Qt::Unchecked){
                  image=cv::imread(ui->FilePatch->text().toStdString());
                  if(image.empty()){
                      QDateTime currentDateTime = QDateTime::currentDateTime();
                      QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                      ui->textEdit_Infor->append(QString("膨胀与腐蚀操作应用:%1，膨胀与腐蚀操作计算失败，文件路径为空").arg(formattedTime));
                      ui->Lab_Image->clear();
                     return;
                  }
               }else if(ui->checkBox_morphoImageVideo->checkState()==Qt::Checked){
                    //先判断当前视频Label
                    if(ui->label_Camera->pixmap()!=nullptr ){
                      const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
                      image=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
                     }
                 }
            }
            //判断是否为空
            if(image.empty()){
                QDateTime currentDateTime = QDateTime::currentDateTime();
                QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                ui->textEdit_Infor->append(QString("开闭操作结果:%1，开闭操作失败，请先按照要求进行图像预处理操作").arg(formattedTime));
                ui->Lab_Image->clear();
                return;
                }
             int operation=0;
             if(ui->checkBox_StarOpenOperation->checkState()==Qt::Checked){
                   operation=cv::MORPH_OPEN;//an opening operation
             }else if(ui->checkBox_StarCloseOperation->checkState()==Qt::Checked){
                   operation=cv::MORPH_CLOSE;//a closing operation
             }
             //获取结构元素
             cv::Mat Kernal=cv::getStructuringElement(ui->spinBox_MorphShape->value(),cv::Size(ui->spinBox_MorphCoreSize->value(),ui->spinBox_MorphCoreSize->value()));
             cv::morphologyEx(image,BasicImageGlobel::OpenCloseOperationResult,operation,Kernal,cv::Point(-1,-1),ui->spinBox_MorphNum->value());
         //开闭操作写入显示区
         ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::OpenCloseOperationResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
         QDateTime currentDateTime = QDateTime::currentDateTime();
         QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
         ui->textEdit_Infor->append(QString("开闭操作结果:%1，开闭操作成功").arg(formattedTime));
     }
}

//形态学梯度
void ImbaProcess::on_checkBox_SartmorphyLogyGrident_stateChanged(int state){
    if(state==Qt::Checked){//是否进行开闭操作
        //判断卷积核是否是奇数，否则加1再进行计算
        if(ui->spinBox_MorphCoreSize->value()%2==0){
           ui->spinBox_MorphCoreSize->setValue(ui->spinBox_MorphCoreSize->value()+1);
        }
         cv::Mat image;
            if(ui->checkBox_UseBinaryResult->checkState()==Qt::Checked){
                if (ui->spinBox_dilateErodeBinaryResult->value()==0){//图像阈值化分割结果
                    image=BasicImageGlobel::ThresholdResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==1){//全阈值化分割结果
                    image=BasicImageGlobel::GlobalThresholdResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==2){//自适应阈值计算结果
                    image=BasicImageGlobel::adaptiveResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==3){//去噪与二值化结果
                    image=BasicImageGlobel::BinaryResult.clone();
                }
            }else if(ui->checkBox_UseBinaryResult->checkState()==Qt::Unchecked){
                //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
                if(ui->checkBox_morphoImageVideo->checkState()==Qt::Unchecked){
                  image=cv::imread(ui->FilePatch->text().toStdString());
                  if(image.empty()){
                      QDateTime currentDateTime = QDateTime::currentDateTime();
                      QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                      ui->textEdit_Infor->append(QString("形态学梯度应用:%1，形态学梯度计算失败，文件路径为空").arg(formattedTime));
                      ui->Lab_Image->clear();
                     return;
                  }
               }else if(ui->checkBox_morphoImageVideo->checkState()==Qt::Checked){
                    //先判断当前视频Label
                    if(ui->label_Camera->pixmap()!=nullptr ){
                      const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
                      image=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
                     }
                 }
            }
            //判断是否为空
            if(image.empty()){
                QDateTime currentDateTime = QDateTime::currentDateTime();
                QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                ui->textEdit_Infor->append(QString("形态学梯度结果:%1，形态学梯度失败，请先按照要求进行图像预处理操作").arg(formattedTime));
                ui->Lab_Image->clear();
                return;
                }
             //获取结构元素
             cv::Mat Kernal=cv::getStructuringElement(ui->spinBox_MorphShape->value(),cv::Size(ui->spinBox_MorphCoreSize->value(),ui->spinBox_MorphCoreSize->value()));
             cv::Mat dilateResult,ErodeResult;
             //膨胀操作
             cv::dilate(image,dilateResult,Kernal,cv::Point(-1,-1),ui->spinBox_MorphNum->value());
             //腐蚀操作
             cv::erode(image,ErodeResult,Kernal,cv::Point(-1,-1),ui->spinBox_MorphNum->value());
                if(ui->spinBox_MorphGrident->value()==0){
                    //基本梯度
                    cv::morphologyEx(image,BasicImageGlobel::GridentResult,cv::MORPH_GRADIENT,Kernal,cv::Point(-1,-1),ui->spinBox_MorphNum->value());
                }else if (ui->spinBox_MorphGrident->value()==1){
                    //内梯度
                    cv::subtract(image,ErodeResult,BasicImageGlobel::GridentResult);
                }else if(ui->spinBox_MorphGrident->value()==2){
                    //外梯度
                    cv::subtract(dilateResult,image,BasicImageGlobel::GridentResult);
                }else if(ui->spinBox_MorphGrident->value()==3){
                    //形态学梯度进行边缘提取
                    if(image.channels()>=3){
                       cv::cvtColor(image,image,cv::COLOR_BGR2GRAY);
                    }
                    cv::morphologyEx(image,BasicImageGlobel::GridentResult,cv::MORPH_GRADIENT,Kernal,cv::Point(-1,-1),ui->spinBox_MorphNum->value());
                    cv::threshold(BasicImageGlobel::GridentResult,BasicImageGlobel::GridentResult,0,255,cv::THRESH_BINARY | cv::THRESH_OTSU);
                }
             //形态学梯度写入显示区
             ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::GridentResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
             QDateTime currentDateTime = QDateTime::currentDateTime();
             QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
             ui->textEdit_Infor->append(QString("形态学梯度结果:%1，形态学梯度成功").arg(formattedTime));
     }
}

//顶帽与黑帽操作
void ImbaProcess::on_checkBox_StarTopBlackheat_stateChanged(int state){
    if(state==Qt::Checked){//是否进行开闭操作
        //判断卷积核是否是奇数，否则加1再进行计算
        if(ui->spinBox_MorphCoreSize->value()%2==0){
           ui->spinBox_MorphCoreSize->setValue(ui->spinBox_MorphCoreSize->value()+1);
        }
         cv::Mat image;
            if(ui->checkBox_UseBinaryResult->checkState()==Qt::Checked){
                if (ui->spinBox_dilateErodeBinaryResult->value()==0){//图像阈值化分割结果
                    image=BasicImageGlobel::ThresholdResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==1){//全阈值化分割结果
                    image=BasicImageGlobel::GlobalThresholdResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==2){//自适应阈值计算结果
                    image=BasicImageGlobel::adaptiveResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==3){//去噪与二值化结果
                    image=BasicImageGlobel::BinaryResult.clone();
                }
            }else if(ui->checkBox_UseBinaryResult->checkState()==Qt::Unchecked){
                //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
                if(ui->checkBox_morphoImageVideo->checkState()==Qt::Unchecked){
                  image=cv::imread(ui->FilePatch->text().toStdString());
                  if(image.empty()){
                      QDateTime currentDateTime = QDateTime::currentDateTime();
                      QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                      ui->textEdit_Infor->append(QString("顶帽与黑帽应用:%1，顶帽与黑失败，文件路径为空").arg(formattedTime));
                      ui->Lab_Image->clear();
                     return;
                  }
               }else if(ui->checkBox_morphoImageVideo->checkState()==Qt::Checked){
                    //先判断当前视频Label
                    if(ui->label_Camera->pixmap()!=nullptr ){
                      const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
                      image=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
                     }
                 }
            }
            //判断是否为空
            if(image.empty()){
                QDateTime currentDateTime = QDateTime::currentDateTime();
                QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                ui->textEdit_Infor->append(QString("顶帽与黑结果:%1，顶帽与黑失败，请先按照要求进行图像预处理操作").arg(formattedTime));
                ui->Lab_Image->clear();
                return;
                }
             //获取结构元素
             cv::Mat Kernal=cv::getStructuringElement(ui->spinBox_MorphShape->value(),cv::Size(ui->spinBox_MorphCoreSize->value(),ui->spinBox_MorphCoreSize->value()));
                if(ui->spinBox_BlackTopHeadSelection->value()==0){
                    //顶帽
                    cv::morphologyEx(image,BasicImageGlobel::BlackTopHeatResult,cv::MORPH_TOPHAT,Kernal,cv::Point(-1,-1),ui->spinBox_MorphNum->value());
                }else if (ui->spinBox_BlackTopHeadSelection->value()==1){
                    //黑帽
                    cv::morphologyEx(image,BasicImageGlobel::BlackTopHeatResult,cv::MORPH_BLACKHAT,Kernal,cv::Point(-1,-1),ui->spinBox_MorphNum->value());
                  }
             //顶帽与黑帽写入显示区
             ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::BlackTopHeatResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
             QDateTime currentDateTime = QDateTime::currentDateTime();
             QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
             ui->textEdit_Infor->append(QString("顶帽与黑帽结果:%1，顶帽与黑帽成功").arg(formattedTime));
     }
}

//击中与击不中操作
void ImbaProcess::on_checkBox_StarHitOrnotHit_stateChanged(int state){
    if(state==Qt::Checked){//是否进行开闭操作
        //判断卷积核是否是奇数，否则加1再进行计算
        if(ui->spinBox_MorphCoreSize->value()%2==0){
           ui->spinBox_MorphCoreSize->setValue(ui->spinBox_MorphCoreSize->value()+1);
        }
         cv::Mat image;
            if(ui->checkBox_UseBinaryResult->checkState()==Qt::Checked){
                if (ui->spinBox_dilateErodeBinaryResult->value()==0){//图像阈值化分割结果
                    image=BasicImageGlobel::ThresholdResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==1){//全阈值化分割结果
                    image=BasicImageGlobel::GlobalThresholdResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==2){//自适应阈值计算结果
                    image=BasicImageGlobel::adaptiveResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==3){//去噪与二值化结果
                    image=BasicImageGlobel::BinaryResult.clone();
                }
            }else if(ui->checkBox_UseBinaryResult->checkState()==Qt::Unchecked){
                //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
                if(ui->checkBox_morphoImageVideo->checkState()==Qt::Unchecked){
                  image=cv::imread(ui->FilePatch->text().toStdString());
                  if(image.empty()){
                      QDateTime currentDateTime = QDateTime::currentDateTime();
                      QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                      ui->textEdit_Infor->append(QString("击中与击不中应用:%1，击中与击不中失败，文件路径为空").arg(formattedTime));
                      ui->Lab_Image->clear();
                     return;
                  }
               }else if(ui->checkBox_morphoImageVideo->checkState()==Qt::Checked){
                    //先判断当前视频Label
                    if(ui->label_Camera->pixmap()!=nullptr ){
                      const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
                      image=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
                     }
                 }
            }
            //判断是否为空
            if(image.empty()){
                QDateTime currentDateTime = QDateTime::currentDateTime();
                QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                ui->textEdit_Infor->append(QString("击中与击不中结果:%1，击中与击不中失败，请先按照要求进行图像预处理操作").arg(formattedTime));
                ui->Lab_Image->clear();
                return;
                }
             //自定义结构结构元素
             int pos1=ui->spinBoxBitMiss_1->value();    int pos21=ui->spinBoxBitMiss_21->value();
             int pos2=ui->spinBoxBitMiss_2->value();    int pos22=ui->spinBoxBitMiss_22->value();
             int pos3=ui->spinBoxBitMiss_3->value();    int pos23=ui->spinBoxBitMiss_23->value();
             int pos4=ui->spinBoxBitMiss_4->value();    int pos24=ui->spinBoxBitMiss_24->value();
             int pos5=ui->spinBoxBitMiss_5->value();    int pos25=ui->spinBoxBitMiss_25->value();
             int pos6=ui->spinBoxBitMiss_6->value();    int pos26=ui->spinBoxBitMiss_26->value();
             int pos7=ui->spinBoxBitMiss_7->value();    int pos27=ui->spinBoxBitMiss_27->value();
             int pos8=ui->spinBoxBitMiss_8->value();    int pos28=ui->spinBoxBitMiss_28->value();
             int pos9=ui->spinBoxBitMiss_9->value();    int pos29=ui->spinBoxBitMiss_29->value();
             if(image.channels()>=3){
                cv::cvtColor(image,image,cv::COLOR_BGR2GRAY);
             }
             cv::Mat Kernel1 =(cv::Mat_<int>(3,3)<<pos1,pos2,pos3,pos4,pos5,pos6,pos7,pos8,pos9);
             cv::Mat Kernel2 =(cv::Mat_<int>(3,3)<<pos21,pos22,pos23,pos24,pos25,pos26,pos27,pos28,pos29);
             cv::Mat Result1,Result2;
             cv::morphologyEx(image,Result1,cv::MORPH_HITMISS,Kernel1,cv::Point(-1,-1),ui->spinBox_MorphNum->value());
             cv::morphologyEx(image,Result2,cv::MORPH_HITMISS,Kernel2,cv::Point(-1,-1),ui->spinBox_MorphNum->value());
             cv::add(Result1,Result2,BasicImageGlobel::BitMissResult);
             //击中与击不中结果写入显示区
             ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::BitMissResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
             QDateTime currentDateTime = QDateTime::currentDateTime();
             QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
             ui->textEdit_Infor->append(QString("击中与击不中结果:%1，击中与击不中成功").arg(formattedTime));
     }
}

//自定义结构性元素操作
void ImbaProcess::on_checkBox_StarCoustmerStruct_stateChanged(int state){
    if(state==Qt::Checked){//是否进行开闭操作
        //判断卷积核是否是奇数，否则加1再进行计算
        if(ui->spinBoxCustomCoreX->value()%2==0){
           ui->spinBoxCustomCoreX->setValue(ui->spinBoxCustomCoreX->value()+1);
        }
        if(ui->spinBoxCustomCoreY->value()%2==0){
           ui->spinBoxCustomCoreY->setValue(ui->spinBoxCustomCoreY->value()+1);
        }
         cv::Mat image;
            if(ui->checkBox_UseBinaryResult->checkState()==Qt::Checked){
                if (ui->spinBox_dilateErodeBinaryResult->value()==0){//图像阈值化分割结果
                    image=BasicImageGlobel::ThresholdResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==1){//全阈值化分割结果
                    image=BasicImageGlobel::GlobalThresholdResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==2){//自适应阈值计算结果
                    image=BasicImageGlobel::adaptiveResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==3){//去噪与二值化结果
                    image=BasicImageGlobel::BinaryResult.clone();
                }
            }else if(ui->checkBox_UseBinaryResult->checkState()==Qt::Unchecked){
                //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
                if(ui->checkBox_morphoImageVideo->checkState()==Qt::Unchecked){
                  image=cv::imread(ui->FilePatch->text().toStdString());
                  if(image.empty()){
                      QDateTime currentDateTime = QDateTime::currentDateTime();
                      QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                      ui->textEdit_Infor->append(QString("自定义结构性元素应用:%1，自定义结构性元素失败，文件路径为空").arg(formattedTime));
                      ui->Lab_Image->clear();
                     return;
                  }
               }else if(ui->checkBox_morphoImageVideo->checkState()==Qt::Checked){
                    //先判断当前视频Label
                    if(ui->label_Camera->pixmap()!=nullptr ){
                      const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
                      image=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
                     }
                 }
            }
            //判断是否为空
            if(image.empty()){
                QDateTime currentDateTime = QDateTime::currentDateTime();
                QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                ui->textEdit_Infor->append(QString("自定义结构性元素结果:%1，自定义结构性元素失败，请先按照要求进行图像预处理操作").arg(formattedTime));
                ui->Lab_Image->clear();
                return;
                }
             //自定义结构结构元素
             if(image.channels()>=3){
                cv::cvtColor(image,image,cv::COLOR_BGR2GRAY);
             }
             //获取结构元素
             if(ui->spinBox_HorVDection->value()==0){//水平或者垂直线段检测
                 cv::Mat Kernel1=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(ui->spinBoxCustomCoreX->value(),ui->spinBoxCustomCoreY->value()));
                 cv::morphologyEx(image,BasicImageGlobel::CustomDectionResult,cv::MORPH_OPEN,Kernel1);
             }else if(ui->spinBox_HorVDection->value()==1){//十字交叉检测
                 if(ui->spinBoxCustomCoreX->value()!=ui->spinBoxCustomCoreY->value()){
                        ui->spinBoxCustomCoreX->setValue(11);
                        ui->spinBoxCustomCoreY->setValue(11);
                  }
                 cv::Mat Kernel2 =cv::getStructuringElement(cv::MORPH_RECT,cv::Size(ui->spinBoxCustomCoreX->value(),ui->spinBoxCustomCoreY->value()));
                 cv::morphologyEx(image,BasicImageGlobel::CustomDectionResult,cv::MORPH_OPEN,Kernel2);

             }
             //击中与击不中结果写入显示区
             ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::CustomDectionResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
             QDateTime currentDateTime = QDateTime::currentDateTime();
             QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
             ui->textEdit_Infor->append(QString("自定义结构元素结果:%1，自定义结构元素成功").arg(formattedTime));
     }
}

//距离变换操作
void ImbaProcess::on_checkBox_StarDistanceChange_stateChanged(int state){
    if(state==Qt::Checked){//是否进行开闭操作
        //判断卷积核是否是奇数，否则加1再进行计算
        if(ui->spinBox_MorphCoreSize->value()%2==0){
           ui->spinBox_MorphCoreSize->setValue(ui->spinBox_MorphCoreSize->value()+1);
        }
         cv::Mat image;
            if(ui->checkBox_UseBinaryResult->checkState()==Qt::Checked){
                if (ui->spinBox_dilateErodeBinaryResult->value()==0){//图像阈值化分割结果
                    image=BasicImageGlobel::ThresholdResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==1){//全阈值化分割结果
                    image=BasicImageGlobel::GlobalThresholdResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==2){//自适应阈值计算结果
                    image=BasicImageGlobel::adaptiveResult.clone();
                }else if(ui->spinBox_dilateErodeBinaryResult->value()==3){//去噪与二值化结果
                    image=BasicImageGlobel::BinaryResult.clone();
                }
            }else if(ui->checkBox_UseBinaryResult->checkState()==Qt::Unchecked){
                //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
                if(ui->checkBox_morphoImageVideo->checkState()==Qt::Unchecked){
                  image=cv::imread(ui->FilePatch->text().toStdString());
                  if(image.empty()){
                      QDateTime currentDateTime = QDateTime::currentDateTime();
                      QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                      ui->textEdit_Infor->append(QString("距离变换应用:%1，距离变换失败，文件路径为空").arg(formattedTime));
                      ui->Lab_Image->clear();
                     return;
                  }
               }else if(ui->checkBox_morphoImageVideo->checkState()==Qt::Checked){
                    //先判断当前视频Label
                    if(ui->label_Camera->pixmap()!=nullptr ){
                      const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
                      image=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
                     }
                 }
            }
            //判断是否为空
            if(image.empty()){
                QDateTime currentDateTime = QDateTime::currentDateTime();
                QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                ui->textEdit_Infor->append(QString("距离变换操作结果:%1，距离变换操作失败，请先按照要求进行图像预处理操作").arg(formattedTime));
                ui->Lab_Image->clear();
                return;
                }
             if(image.channels()>=3){
                cv::cvtColor(image,image,cv::COLOR_BGR2GRAY);
             }
             cv::distanceTransform(image,BasicImageGlobel::CustomDectionResult,ui->spinBox_DistanceType->value(),3);
             cv::normalize(BasicImageGlobel::CustomDectionResult,BasicImageGlobel::CustomDectionResult,0,255,cv::NORM_MINMAX);
             BasicImageGlobel::CustomDectionResult.convertTo(BasicImageGlobel::CustomDectionResult,CV_8U);
             //击中与击不中结果写入显示区
             ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::CustomDectionResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
             QDateTime currentDateTime = QDateTime::currentDateTime();
             QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
             ui->textEdit_Infor->append(QString("自定义结构元素结果:%1，自定义结构元素成功").arg(formattedTime));
     }
}

//分水岭分割操作
void ImbaProcess::on_checkBox_StarWaterShed_stateChanged(int state){



}


//图像金字塔
void ImbaProcess::on_checkBox_ImagePyramid_stateChanged(int state){
    //图像金字塔选中计算
   if(state==Qt::Checked){
       //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
       cv::Mat image;
       if(ui->checkBox_PyramidImageVideo->checkState()==Qt::Unchecked){
           image=cv::imread(ui->FilePatch->text().toStdString());
         }
       else if (ui->checkBox_PyramidImageVideo->checkState()==Qt::Checked){
           //先判断当前视频Label
           if(ui->label_Camera->pixmap()!=nullptr ){
             const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
             image=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
           }
         }
       if(image.empty()){
           QDateTime currentDateTime = QDateTime::currentDateTime();
           QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
           ui->textEdit_Infor->append(QString("本地图片数据应用:%1，图像金字塔失败，文件路径为空").arg(formattedTime));
           ui->Lab_Image->clear();
           return;
          }
          //图像金字塔
          cv::Mat temp=image.clone();
          if(ui->spinBox_PyramidMode->value()==0){//金字塔缩放
              for(int i=0;i<ui->spinBox_PyramidNum->value();i++ ) {
                  cv::Mat dst;
                  cv::pyrDown(temp,dst);
                  cv::imshow(cv::format("Reduce:%d",(i+1)),dst);
                  dst.copyTo(temp);
                  BasicImageGlobel::ImagePyramidResult.push_back(dst);
              }

          }else if(ui->spinBox_PyramidMode->value()==1){//金字塔扩展
              for(int i=0;i<ui->spinBox_PyramidNum->value();i++ ) {
                  cv::Mat dst;
                  cv::pyrUp(temp,dst);
                  cv::imshow(cv::format("Expand:%d",(i+1)),dst);
                  dst.copyTo(temp);
                  BasicImageGlobel::ImagePyramidResult.push_back(dst);
              }
          }
          //图像金字塔结果写入显示区
          ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::ImagePyramidResult[0]).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
          QDateTime currentDateTime = QDateTime::currentDateTime();
          QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
          ui->textEdit_Infor->append(QString("图像金字塔结果:%1，图像金字塔成功").arg(formattedTime));
     }
}

//Harris角点检测
void ImbaProcess::on_checkBox_HarrisDection_stateChanged(int state){
    //Harris角点检测选中计算
   if(state==Qt::Checked){
       //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
       cv::Mat image;
       if(ui->checkBox_PyramidImageVideo->checkState()==Qt::Unchecked){
           image=cv::imread(ui->FilePatch->text().toStdString());
         }
       else if (ui->checkBox_PyramidImageVideo->checkState()==Qt::Checked){
           //先判断当前视频Label
           if(ui->label_Camera->pixmap()!=nullptr ){
             const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
             image=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
           }
         }
       if(image.empty()){
           QDateTime currentDateTime = QDateTime::currentDateTime();
           QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
           ui->textEdit_Infor->append(QString("本地图片数据应用:%1，Harris角点检测失败，文件路径为空").arg(formattedTime));
           ui->Lab_Image->clear();
           return;
          }
          BasicImageGlobel::HarrisCornerResult=image.clone();
          cv::Mat Gray,dst;
          cv::cvtColor(image,Gray,cv::COLOR_BGR2GRAY);
          //Harris角点检测
          cv::cornerHarris(Gray,dst,ui->spinBox_HarrisBlockSize->value(),ui->spinBox_HarrisGridentSize->value(),ui->doubleSpinBox_HarrisSensitivity->value());
          //结果归一化
          cv::Mat Black=cv::Mat::zeros(dst.size(),dst.type());
          cv::normalize(dst,Black,0,255,cv::NORM_MINMAX);
          //取出绝对值，并转换为8为无符号类型
          cv::convertScaleAbs(Black,Black);

          //绘制角点
          cv::RNG rng=cv::RNG(QTime::currentTime().msec());//设置随机数种子
          for (int i=0;i<Black.rows;i++){
              for (int j=0;j<Black.cols;j++) {
                   int pix=Black.at<uchar>(i,j);
                   //阈值判定
                   if(pix>ui->spinBox_HarrisThreshold->value()){
                      int b=rng.uniform(0,255);
                      int g=rng.uniform(0,255);
                      int r=rng.uniform(0,255);
                      cv::circle(BasicImageGlobel::HarrisCornerResult,cv::Point(j,i),2,cv::Scalar(b,g,r),2,8);
                   }
              }
          }

          //Harris角点检测结果写入显示区
          ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::HarrisCornerResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
          QDateTime currentDateTime = QDateTime::currentDateTime();
          QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
          ui->textEdit_Infor->append(QString("Harris角点检测结果:%1，Harris角点检测成功").arg(formattedTime));
     }
}
//shi-tomas角点检测
void ImbaProcess::on_checkBox_shiTomasDection_stateChanged(int state){
    //shi-tomas角点检测选中计算
   if(state==Qt::Checked){
       //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
       cv::Mat image;
       if(ui->checkBox_PyramidImageVideo->checkState()==Qt::Unchecked){
           image=cv::imread(ui->FilePatch->text().toStdString());
         }
       else if (ui->checkBox_PyramidImageVideo->checkState()==Qt::Checked){
           //先判断当前视频Label
           if(ui->label_Camera->pixmap()!=nullptr ){
             const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
             image=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
           }
         }
       if(image.empty()){
           QDateTime currentDateTime = QDateTime::currentDateTime();
           QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
           ui->textEdit_Infor->append(QString("本地图片数据应用:%1，shi-tomas角点检测失败，文件路径为空").arg(formattedTime));
           ui->Lab_Image->clear();
           return;
          }
          BasicImageGlobel::shiTomasCornerResult=image.clone();
          cv::Mat Gray,dst;
          cv::cvtColor(image,Gray,cv::COLOR_BGR2GRAY);
          //shiTomas角点检测
          std::vector<cv::Point>Cornel;
          cv::goodFeaturesToTrack(Gray,Cornel,ui->spinBox_MaxPointNumber->value(),ui->doubleSpinBox_shiTomasSensitivity->value(),ui->spinBox_shiTomasMinDistance->value(),cv::Mat(),ui->spinBox_shiTomasMaxPointNumber->value());
          //绘制角点
          cv::RNG rng=cv::RNG(QTime::currentTime().msec());//设置随机数种子
          for (int i=0;i<Cornel.size();i++){
                      cv::Point pt=Cornel[i];
                      int b=rng.uniform(0,255);
                      int g=rng.uniform(0,255);
                      int r=rng.uniform(0,255);
                      cv::circle(BasicImageGlobel::shiTomasCornerResult,pt,2,cv::Scalar(b,g,r),2,8);
            }
          //shi-tomas角点检测结果写入显示区
          ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::shiTomasCornerResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
          QDateTime currentDateTime = QDateTime::currentDateTime();
          QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
          ui->textEdit_Infor->append(QString("shi-tomas角点检测结果:%1，shi-tomas角点检测成功").arg(formattedTime));
     }
}
//HOG特征描述子
void ImbaProcess::on_checkBox_HogFeature_stateChanged(int state){
    //HOG特征描述子检测选中计算
   if(state==Qt::Checked){
       //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
       cv::Mat image;
       if(ui->checkBox_PyramidImageVideo->checkState()==Qt::Unchecked){
           image=cv::imread(ui->FilePatch->text().toStdString());
         }
       else if (ui->checkBox_PyramidImageVideo->checkState()==Qt::Checked){
           //先判断当前视频Label
           if(ui->label_Camera->pixmap()!=nullptr ){
             const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
             image=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
           }
         }
       if(image.empty()){
           QDateTime currentDateTime = QDateTime::currentDateTime();
           QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
           ui->textEdit_Infor->append(QString("本地图片数据应用:%1，HOG特征描述子检测失败，文件路径为空").arg(formattedTime));
           ui->Lab_Image->clear();
           return;
          }
          //HOG特征描述子计算
          BasicImageGlobel::HOGfeatureDescripResult=image.clone();
          cv::Mat Gray;
          cv::cvtColor(image,Gray,cv::COLOR_BGR2GRAY);
          std::vector<float> descriptors;
          int WinSize=ui->spinBox_WinSize->value();
          std::unique_ptr<cv::HOGDescriptor> hogdescriptor = std::make_unique<cv::HOGDescriptor>();
          hogdescriptor->compute(Gray,descriptors,cv::Size(WinSize,WinSize));
          ui->lineEdit_HOGDescriptorsNumber->setText(QString().setNum(descriptors.size()));
          //HOG特征行人检测
          hogdescriptor->setSVMDetector(hogdescriptor->getDefaultPeopleDetector());
          std::vector<cv::Rect>object;
          double threshold=ui->doubleSpinBox_PeopleDectionThredhold->value();
          cv::Size Winsize=cv::Size(ui->spinBox_WinSize->value(),ui->spinBox_WinSize->value());
          cv::Size WinPading=cv::Size(ui->spinBox_DectionArea->value(),ui->spinBox_DectionArea->value());
          double PyramidRatio=ui->doubleSpinBox_PyramidRatio->value();
          hogdescriptor->detectMultiScale(BasicImageGlobel::HOGfeatureDescripResult,object,threshold,Winsize,WinPading,PyramidRatio);
          for (int i=0;i<object.size();i++ ) {
              cv::rectangle(BasicImageGlobel::HOGfeatureDescripResult,object[i],cv::Scalar(0,0,255),2,8,0);
          }
          //HOG特征描述子检测结果写入显示区
          ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::HOGfeatureDescripResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
          QDateTime currentDateTime = QDateTime::currentDateTime();
          QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
          ui->textEdit_Infor->append(QString("HOG特征描述子检测结果:%1，HOG特征描述子检测成功").arg(formattedTime));
     }
}
//ORB特征描述子
void ImbaProcess::on_BtnORBTemplateobject_clicked()
{
    //待查找对象路径
    QString lastPath = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    QString fileName = QFileDialog::getOpenFileName(this, "请选择图片", lastPath, "图片(*.png *.jpg);;");
    if(fileName.isEmpty())
    {
        //加载失败，向信息区中写入信息
        QDateTime currentDateTime = QDateTime::currentDateTime();
        QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
        ui->textEdit_Infor->append(QString("本地图片数据采集:%1，图像加载失败，文件路径为空").arg(formattedTime));
        ui->Lab_Image->clear();
        ui->FilePatch->clear();
        return;
    }
    //将路径加载至文件路径显示栏中
    ui->FilePatch_ORBTemplateobject->setText(fileName);
}

void ImbaProcess::on_checkBox_ORBFratureDection_stateChanged(int state){
    //ORB特征描述子检测选中计算
   if(state==Qt::Checked){
       //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
       cv::Mat imageTemplata,imageObject;
       if(ui->checkBox_PyramidImageVideo->checkState()==Qt::Unchecked){
           imageTemplata=cv::imread(ui->FilePatch_ORBTemplateobject->text().toStdString());
           imageObject=cv::imread(ui->FilePatch->text().toStdString());
         }
       else if (ui->checkBox_PyramidImageVideo->checkState()==Qt::Checked){
           //先判断当前视频Label
           if(ui->label_Camera->pixmap()!=nullptr ){
             const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
             imageObject=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
           }
         }
       if(imageObject.empty()||imageTemplata.empty()){
           QDateTime currentDateTime = QDateTime::currentDateTime();
           QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
           ui->textEdit_Infor->append(QString("本地图片数据应用:%1，ORB特征描述子检测失败，文件路径为空").arg(formattedTime));
           ui->Lab_Image->clear();
           return;
          }
          //创建ORB描述子(将检测对象和模板对象的描述子和关键点找到)
          cv::Ptr<cv::ORB>orb_detectoe=cv::ORB::create(ui->spinBox_MaxfeatureNumber->value());
          std::vector<cv::KeyPoint>TempKeyPoing;
          std::vector<cv::KeyPoint>ObjectKeyPoing;
          cv::Mat TempDescrip,ObjectDescrip;
          orb_detectoe->detectAndCompute(imageTemplata,cv::Mat(),TempKeyPoing,TempDescrip);
          orb_detectoe->detectAndCompute(imageObject,cv::Mat(),ObjectKeyPoing,ObjectDescrip);
          //暴力匹配或者FLANN匹配
          if(ui->spinBox_ORBMatchingMode->value()==0){//暴力匹配
                //创建暴力匹配
                cv::Ptr<cv::BFMatcher>bfmatcher=cv::BFMatcher::create(cv::NORM_HAMMING,false);
                //暴力匹配结果容器(将模板对象和检测对象的两个关键点和描述子进行匹配，计算出匹配结果并绘制出来)
                std::vector<cv::DMatch>matches;
                bfmatcher->match(ObjectDescrip,TempDescrip,matches);
                cv::drawMatches(imageTemplata,TempKeyPoing,imageObject,ObjectKeyPoing,matches,BasicImageGlobel::ORBMatchingResult,cv::Scalar(255,0,0));
          }else if(ui->spinBox_ORBMatchingMode->value()==1){//FLANN匹配
                //创建Flann匹配
                cv::FlannBasedMatcher flannMatcher=cv::FlannBasedMatcher(new cv::flann::LshIndexParams(6,12,2));
                std::vector<cv::DMatch>matches;
                //FLANN匹配结果容器(将模板对象和检测对象的两个关键点和描述子进行匹配，计算出匹配结果并绘制出来)
                flannMatcher.match(ObjectDescrip,TempDescrip,matches);
                cv::drawMatches(imageTemplata,TempKeyPoing,imageObject,ObjectKeyPoing,matches,BasicImageGlobel::ORBMatchingResult,cv::Scalar(255,0,0));
          }
          //ORB特征描述子检测结果写入显示区
          ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::ORBMatchingResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
          QDateTime currentDateTime = QDateTime::currentDateTime();
          QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
          ui->textEdit_Infor->append(QString("ORB特征描述子检测结果:%1，ORB特征描述子检测成功").arg(formattedTime));
     }
}
//基于特征的对象检测
void ImbaProcess::on_checkBox_BasicFeatureDection_stateChanged(int state){
    //基于特征的对象检测选中计算
   if(state==Qt::Checked){
       //判断是加载图像还是视频(不选中就加载图像，选中就加载视频)
       cv::Mat imageTemplata,imageObject;
       if(ui->checkBox_PyramidImageVideo->checkState()==Qt::Unchecked){
           imageTemplata=cv::imread(ui->FilePatch_ORBTemplateobject->text().toStdString());
           imageObject=cv::imread(ui->FilePatch->text().toStdString());
         }
       else if (ui->checkBox_PyramidImageVideo->checkState()==Qt::Checked){
           //先判断当前视频Label
           if(ui->label_Camera->pixmap()!=nullptr ){
             const QPixmap *pixmap = ui->label_Camera->pixmap();//将视频区的图像加载转换出来
             imageObject=QPixmapToMat(*pixmap);//将加载出来的Pixmap对象转换为Mat对象原图
           }
         }
       if(imageObject.empty()||imageTemplata.empty()){
           QDateTime currentDateTime = QDateTime::currentDateTime();
           QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
           ui->textEdit_Infor->append(QString("本地图片数据应用:%1，基于特征的对象检测失败，文件路径为空").arg(formattedTime));
           ui->Lab_Image->clear();
           return;
          }
          //创建ORB描述子(将检测对象和模板对象的描述子和关键点找到)
          cv::Ptr<cv::ORB>orb_detectoe=cv::ORB::create(ui->spinBox_MaxfeatureNumber->value());
          std::vector<cv::KeyPoint>TempKeyPoing;
          std::vector<cv::KeyPoint>ObjectKeyPoing;
          cv::Mat TempDescrip,ObjectDescrip;
          orb_detectoe->detectAndCompute(imageTemplata,cv::Mat(),TempKeyPoing,TempDescrip);
          orb_detectoe->detectAndCompute(imageObject,cv::Mat(),ObjectKeyPoing,ObjectDescrip);
          //创建暴力匹配
          cv::Ptr<cv::BFMatcher>bfmatcher=cv::BFMatcher::create(cv::NORM_HAMMING,false);
          //暴力匹配结果容器(将模板对象和检测对象的两个关键点和描述子进行匹配，计算出匹配结果并绘制出来)
          std::vector<cv::DMatch>matches;
          bfmatcher->match(TempDescrip,ObjectDescrip,matches);
          //筛选出优质的暴力匹配点
          std::sort(matches.begin(),matches.end());
          const int NumgoodMatches=matches.size()*((double)ui->spinBox_NumGoodMatches->value()/100.0);
          //清除不优质的点位
          matches.erase(matches.begin()+NumgoodMatches,matches.end());
          cv::drawMatches(imageTemplata,TempKeyPoing,imageObject,ObjectKeyPoing,matches,BasicImageGlobel::ObjectDectionResult,cv::Scalar(255,0,0));
          //求解单应性矩阵
          std::vector<cv::Point2f>Temp_Pts;
          std::vector<cv::Point2f>Object_Pts;
          for (int i=0;i<matches.size();i++) {
              Temp_Pts.push_back(TempKeyPoing[matches[i].queryIdx].pt);
              Object_Pts.push_back(ObjectKeyPoing[matches[i].trainIdx].pt);
          }
          if(ui->spinBox_HRectFunction->value()==12){
              ui->spinBox_HRectFunction->setValue(ui->spinBox_HRectFunction->value()+4);
          }
          cv::Mat H=cv::findHomography(Temp_Pts,Object_Pts,cv::RANSAC);
          //根据变换矩阵得到目标点对
          std::vector<cv::Point2f>TempCorners(4);
          TempCorners[0]=cv::Point2f(0,0);
          TempCorners[1]=cv::Point2f(imageTemplata.cols,0);
          TempCorners[2]=cv::Point2f(imageTemplata.cols,imageTemplata.rows);
          TempCorners[3]=cv::Point2f(0,imageTemplata.rows);
          //透视变换
          std::vector<cv::Point2f>ObjectCorners(4);
          cv::perspectiveTransform(TempCorners,ObjectCorners,H);
          //绘制结果
          cv::line(BasicImageGlobel::ObjectDectionResult,TempCorners[0],TempCorners[1],cv::Scalar(0,0,255),4);//模板
          cv::line(BasicImageGlobel::ObjectDectionResult,TempCorners[1],TempCorners[2],cv::Scalar(0,0,255),4);
          cv::line(BasicImageGlobel::ObjectDectionResult,TempCorners[2],TempCorners[3],cv::Scalar(0,0,255),4);
          cv::line(BasicImageGlobel::ObjectDectionResult,TempCorners[3],TempCorners[0],cv::Scalar(0,0,255),4);

          cv::line(BasicImageGlobel::ObjectDectionResult,ObjectCorners[0]+cv::Point2f(imageTemplata.cols,0),ObjectCorners[1]+cv::Point2f(imageTemplata.cols,0),cv::Scalar(0,0,255),4);//查找对象
          cv::line(BasicImageGlobel::ObjectDectionResult,ObjectCorners[1]+cv::Point2f(imageTemplata.cols,0),ObjectCorners[2]+cv::Point2f(imageTemplata.cols,0),cv::Scalar(0,0,255),4);
          cv::line(BasicImageGlobel::ObjectDectionResult,ObjectCorners[2]+cv::Point2f(imageTemplata.cols,0),ObjectCorners[3]+cv::Point2f(imageTemplata.cols,0),cv::Scalar(0,0,255),4);
          cv::line(BasicImageGlobel::ObjectDectionResult,ObjectCorners[3]+cv::Point2f(imageTemplata.cols,0),ObjectCorners[0]+cv::Point2f(imageTemplata.cols,0),cv::Scalar(0,0,255),4);

          //基于特征的对象检测结果写入显示区
          ui->Lab_Image->setPixmap(matToQPixmap(BasicImageGlobel::ObjectDectionResult).scaled(ui->Lab_Image->size()));//计算后返回并设置到Qlabel中
          QDateTime currentDateTime = QDateTime::currentDateTime();
          QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
          ui->textEdit_Infor->append(QString("基于特征的对象检测结果:%1，基于特征的对象检测成功").arg(formattedTime));
     }
}


//基于颜色对象跟踪
void ImbaProcess::on_checkBox_ColorTailAfter_stateChanged(int state){
  cv::VideoCapture capture;
  //基于颜色对象跟踪选中计算
  if(state==Qt::Checked){
      BasicImageGlobel::shouldStop=1;
        if(ui->checkBox_VideoAnalysLocalfile->checkState()==Qt::Checked){//选择加载本地视频
            QString lastVideoPath = QFileDialog::getOpenFileName(this, "请选择视频", QStandardPaths::writableLocation(QStandardPaths::MoviesLocation), "视频(*.avi *.mp4);;");
            if(lastVideoPath.isEmpty())
            {
                //停止数据采集，向信息区中写入信息
                QDateTime currentDateTime = QDateTime::currentDateTime();
                QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                ui->textEdit_Infor->append(QString("本地视频数据采集:%1，视频数据采集失败，路径为空").arg(formattedTime));
                return;
            }
           bool ret1 =capture.open(lastVideoPath.toStdString());
           int height =capture.get(cv::CAP_PROP_FRAME_HEIGHT);//视频高
           int width  =capture.get(cv::CAP_PROP_FRAME_WIDTH);//视频宽
           double fps =capture.get(cv::CAP_PROP_FPS);//视频帧率
           double count =capture.get(cv::CAP_PROP_FRAME_COUNT);//总帧率
           //在信息区显示当前加载视频的信息格式
           QDateTime currentDateTime = QDateTime::currentDateTime();
           QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
           ui->textEdit_Infor->append(QString("实时相机数据采集:%1，Height:%2,Width:%3,FPS:%4,Count:%5").arg(formattedTime).arg(height).arg(width).arg(fps).arg(count));
        }else if(ui->checkBox_VideoAnalysCameraData->checkState()==Qt::Checked){//选择加载相机数据
           bool ret2 =capture.open(ui->spinBox_VideoAnalysCameraNum->value());
           int height =capture.get(cv::CAP_PROP_FRAME_HEIGHT);//视频高
           int width  =capture.get(cv::CAP_PROP_FRAME_WIDTH);//视频宽
           double fps =capture.get(cv::CAP_PROP_FPS);//视频帧率
           double count =capture.get(cv::CAP_PROP_FRAME_COUNT);//总帧率
           //在信息区显示当前加载视频的信息格式
           QDateTime currentDateTime = QDateTime::currentDateTime();
           QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
           ui->textEdit_Infor->append(QString("实时相机数据采集:%1，Height:%2,Width:%3,FPS:%4,Count:%5").arg(formattedTime).arg(height).arg(width).arg(fps).arg(count));
        }
    //相机每帧图像
    cv::Mat frame;
    while(BasicImageGlobel::shouldStop){
       capture.read(frame);
       if(frame.empty()){
           break;
         }
       cv::Mat hsvimage;
       cv::cvtColor(frame,hsvimage,cv::COLOR_BGR2HSV);//转换为HSV格式
       cv::Mat se=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
       //提取目标颜色区域
       int hmin=ui->spinBox_HminvalueVideo->value();
       int smin=ui->spinBox_SminvalueVideo->value();
       int vmin=ui->spinBox_VminvalueVideo->value();
       int hmax=ui->spinBox_HmaxvalueVideo->value();
       int smax=ui->spinBox_SmaxvalueVideo->value();
       int vmax=ui->spinBox_VmaxvalueVideo->value();
       cv::Mat mask;
       cv::inRange(hsvimage,cv::Scalar(hmin,smin,vmin),cv::Scalar(hmax,smax,vmax),mask);
       //对Mask区域进行形态学处理,并进行轮廓发现
       cv::morphologyEx(mask,mask,cv::MORPH_CLOSE,se);
       std::vector<std::vector<cv::Point>>Contours;
       std::vector<cv::Vec4i> Hierarchy;
       cv::findContours(mask,Contours,Hierarchy,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
       for (int i=0;i<Contours.size();i++) {
           //像素值滤波
           if(Contours[i].size()>ui->spinBox_VideoTargetFilting->value()){
                //轮廓点绘制最小外接矩形
                cv::Rect box=cv::boundingRect(Contours[i]);
                cv::rectangle(frame,box,cv::Scalar(0,255,0),2,8);
                //绘制被检测点圆心
                cv::RotatedRect Elli=cv::fitEllipse(Contours[i]);
                cv::circle(frame,Elli.center,2,cv::Scalar(255,0,0),2,8);
           }
       }
       //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
       QPixmap pix = matToQPixmap(frame);
       pix.scaled(ui->Lab_Image->size(),Qt::KeepAspectRatio);
       ui->Lab_Image->setScaledContents(true);
       ui->Lab_Image->setPixmap(pix);
       cv::waitKey(100);
      }
     ui->Lab_Image->clear();
     capture.release();
  }else if(state==Qt::Unchecked){
      BasicImageGlobel::shouldStop=0;
      capture.release();//释放相机数据
  }
}
//视频背景分析
void ImbaProcess::on_checkBox_VideoBlackGroundAnalyse_stateChanged(int state){
    cv::VideoCapture capture;
    //基于颜色对象跟踪选中计算
    if(state==Qt::Checked){
        BasicImageGlobel::shouldStop=1;
          if(ui->checkBox_VideoAnalysLocalfile->checkState()==Qt::Checked){//选择加载本地视频
              QString lastVideoPath = QFileDialog::getOpenFileName(this, "请选择视频", QStandardPaths::writableLocation(QStandardPaths::MoviesLocation), "视频(*.avi *.mp4);;");
              if(lastVideoPath.isEmpty())
              {
                  //停止数据采集，向信息区中写入信息
                  QDateTime currentDateTime = QDateTime::currentDateTime();
                  QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                  ui->textEdit_Infor->append(QString("本地视频数据采集:%1，视频背景分析失败，路径为空").arg(formattedTime));
                  return;
              }
             bool ret1 =capture.open(lastVideoPath.toStdString());
             int height =capture.get(cv::CAP_PROP_FRAME_HEIGHT);//视频高
             int width  =capture.get(cv::CAP_PROP_FRAME_WIDTH);//视频宽
             double fps =capture.get(cv::CAP_PROP_FPS);//视频帧率
             double count =capture.get(cv::CAP_PROP_FRAME_COUNT);//总帧率
             //在信息区显示当前加载视频的信息格式
             QDateTime currentDateTime = QDateTime::currentDateTime();
             QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
             ui->textEdit_Infor->append(QString("实时相机数据采集:%1，Height:%2,Width:%3,FPS:%4,Count:%5").arg(formattedTime).arg(height).arg(width).arg(fps).arg(count));
          }else if(ui->checkBox_VideoAnalysCameraData->checkState()==Qt::Checked){//选择加载相机数据
             bool ret2 =capture.open(ui->spinBox_VideoAnalysCameraNum->value());
             int height =capture.get(cv::CAP_PROP_FRAME_HEIGHT);//视频高
             int width  =capture.get(cv::CAP_PROP_FRAME_WIDTH);//视频宽
             double fps =capture.get(cv::CAP_PROP_FPS);//视频帧率
             double count =capture.get(cv::CAP_PROP_FRAME_COUNT);//总帧率
             //在信息区显示当前加载视频的信息格式
             QDateTime currentDateTime = QDateTime::currentDateTime();
             QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
             ui->textEdit_Infor->append(QString("实时相机数据采集:%1，Height:%2,Width:%3,FPS:%4,Count:%5").arg(formattedTime).arg(height).arg(width).arg(fps).arg(count));
          }
          //创建背景，提取MOG2对象
          cv::Ptr<cv::BackgroundSubtractorMOG2> pMog2=cv::createBackgroundSubtractorMOG2(ui->spinBox_HistoryLenthHis->value(),ui->spinBox_PregroundThreshold->value());
          //相机每帧图像
          cv::Mat frame,mask,black_image;
          while(BasicImageGlobel::shouldStop){
             capture.read(frame);
             if(frame.empty()){
                 break;
               }
             //提取前景
             pMog2->apply(frame,mask);
             //提取背景
             pMog2->getBackgroundImage(black_image);

             //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
             QPixmap pix = matToQPixmap(mask);
             pix.scaled(ui->Lab_Image->size(),Qt::KeepAspectRatio);
             ui->Lab_Image->setScaledContents(true);
             ui->Lab_Image->setPixmap(pix);
             cv::waitKey(10);
            }
           ui->Lab_Image->clear();
           capture.release();
        }else if(state==Qt::Unchecked){
            BasicImageGlobel::shouldStop=0;
            capture.release();//释放相机数据
        }
}
//基于帧差法分析
void ImbaProcess::on_checkBox_FramDiferenceAnalyse_stateChanged(int state){
    cv::VideoCapture capture;
    //基于颜色对象跟踪选中计算
    if(state==Qt::Checked){
        BasicImageGlobel::shouldStop=1;
          if(ui->checkBox_VideoAnalysLocalfile->checkState()==Qt::Checked){//选择加载本地视频
              QString lastVideoPath = QFileDialog::getOpenFileName(this, "请选择视频", QStandardPaths::writableLocation(QStandardPaths::MoviesLocation), "视频(*.avi *.mp4);;");
              if(lastVideoPath.isEmpty())
              {
                  //停止数据采集，向信息区中写入信息
                  QDateTime currentDateTime = QDateTime::currentDateTime();
                  QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                  ui->textEdit_Infor->append(QString("本地视频数据采集:%1，视频背景分析失败，路径为空").arg(formattedTime));
                  return;
              }
             bool ret1 =capture.open(lastVideoPath.toStdString());
             int height =capture.get(cv::CAP_PROP_FRAME_HEIGHT);//视频高
             int width  =capture.get(cv::CAP_PROP_FRAME_WIDTH);//视频宽
             double fps =capture.get(cv::CAP_PROP_FPS);//视频帧率
             double count =capture.get(cv::CAP_PROP_FRAME_COUNT);//总帧率
             //在信息区显示当前加载视频的信息格式
             QDateTime currentDateTime = QDateTime::currentDateTime();
             QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
             ui->textEdit_Infor->append(QString("实时相机数据采集:%1，Height:%2,Width:%3,FPS:%4,Count:%5").arg(formattedTime).arg(height).arg(width).arg(fps).arg(count));
          }else if(ui->checkBox_VideoAnalysCameraData->checkState()==Qt::Checked){//选择加载相机数据
             bool ret2 =capture.open(ui->spinBox_VideoAnalysCameraNum->value());
             int height =capture.get(cv::CAP_PROP_FRAME_HEIGHT);//视频高
             int width  =capture.get(cv::CAP_PROP_FRAME_WIDTH);//视频宽
             double fps =capture.get(cv::CAP_PROP_FPS);//视频帧率
             double count =capture.get(cv::CAP_PROP_FRAME_COUNT);//总帧率
             //在信息区显示当前加载视频的信息格式
             QDateTime currentDateTime = QDateTime::currentDateTime();
             QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
             ui->textEdit_Infor->append(QString("实时相机数据采集:%1，Height:%2,Width:%3,FPS:%4,Count:%5").arg(formattedTime).arg(height).arg(width).arg(fps).arg(count));
          }
          //截取前一帧图像，并将其转换为灰度图像
          cv::Mat Prefram,Pregray;
          capture.read(Prefram);
          cv::cvtColor(Prefram,Pregray,cv::COLOR_BGR2GRAY);
          cv::GaussianBlur(Pregray,Pregray,cv::Size(3,3),15);
          //获取结构性元素
          cv::Mat se=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3));
          //相机每帧图像
          cv::Mat frame,framegray,binary;
          while(BasicImageGlobel::shouldStop){
             capture.read(frame);
             if(frame.empty())break;
               //当前帧图像减去前一帧图像
               cv::cvtColor(frame,framegray,cv::COLOR_BGR2GRAY);
               cv::GaussianBlur(framegray,framegray,cv::Size(3,3),15);
               cv::subtract(framegray,Pregray,binary);
               //二值化并执行开操作
               cv::threshold(binary,binary,127,255,cv::THRESH_BINARY|cv::THRESH_OTSU);
               cv::morphologyEx(binary,binary,cv::MORPH_OPEN,se);
             //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
             QPixmap pix = matToQPixmap(binary);
             pix.scaled(ui->Lab_Image->size(),Qt::KeepAspectRatio);
             ui->Lab_Image->setScaledContents(true);
             ui->Lab_Image->setPixmap(pix);
             //将前一帧图像更新
             framegray.copyTo(Pregray);
             cv::waitKey(1);
            }
           ui->Lab_Image->clear();
           capture.release();
        }else if(state==Qt::Unchecked){
            BasicImageGlobel::shouldStop=0;
            capture.release();//释放相机数据
        }
}
//稀疏光流绘制shitomas角点
void draw_goodFeatures(cv::Mat &image, std::vector<cv::Point2f> goodFeatures) {
    for (size_t t = 0; t < goodFeatures.size(); t++) {
        circle(image, goodFeatures[t], 2, cv::Scalar(0, 255, 0), 2, 8, 0);
    }
}
//绘制线段
void draw_lines(cv::Mat &image, std::vector<cv::Point2f> pt1, std::vector<cv::Point2f> pt2) {
    cv::RNG rng(12345);
    std::vector<cv::Scalar> color_lut;
    if (color_lut.size() < pt1.size()) {
        for (size_t t = 0; t < pt1.size(); t++) {
            color_lut.push_back(cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
        }
    }
    for (size_t t = 0; t < pt1.size(); t++) {
        line(image, pt1[t], pt2[t], color_lut[t], 2, 8, 0);
    }
}
void ImbaProcess::on_checkBox_SparseOpticalFlow_stateChanged(int state){
    cv::VideoCapture capture;
    //基于颜色对象跟踪选中计算
    if(state==Qt::Checked){
        BasicImageGlobel::shouldStop=1;
          if(ui->checkBox_VideoAnalysLocalfile->checkState()==Qt::Checked){//选择加载本地视频
              QString lastVideoPath = QFileDialog::getOpenFileName(this, "请选择视频", QStandardPaths::writableLocation(QStandardPaths::MoviesLocation), "视频(*.avi *.mp4);;");
              if(lastVideoPath.isEmpty())
              {
                  //停止数据采集，向信息区中写入信息
                  QDateTime currentDateTime = QDateTime::currentDateTime();
                  QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                  ui->textEdit_Infor->append(QString("本地视频数据采集:%1，视频背景分析失败，路径为空").arg(formattedTime));
                  return;
              }
             bool ret1 =capture.open(lastVideoPath.toStdString());
             int height =capture.get(cv::CAP_PROP_FRAME_HEIGHT);//视频高
             int width  =capture.get(cv::CAP_PROP_FRAME_WIDTH);//视频宽
             double fps =capture.get(cv::CAP_PROP_FPS);//视频帧率
             double count =capture.get(cv::CAP_PROP_FRAME_COUNT);//总帧率
             //在信息区显示当前加载视频的信息格式
             QDateTime currentDateTime = QDateTime::currentDateTime();
             QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
             ui->textEdit_Infor->append(QString("实时相机数据采集:%1，Height:%2,Width:%3,FPS:%4,Count:%5").arg(formattedTime).arg(height).arg(width).arg(fps).arg(count));
          }else if(ui->checkBox_VideoAnalysCameraData->checkState()==Qt::Checked){//选择加载相机数据
             bool ret2 =capture.open(ui->spinBox_VideoAnalysCameraNum->value());
             int height =capture.get(cv::CAP_PROP_FRAME_HEIGHT);//视频高
             int width  =capture.get(cv::CAP_PROP_FRAME_WIDTH);//视频宽
             double fps =capture.get(cv::CAP_PROP_FPS);//视频帧率
             double count =capture.get(cv::CAP_PROP_FRAME_COUNT);//总帧率
             //在信息区显示当前加载视频的信息格式
             QDateTime currentDateTime = QDateTime::currentDateTime();
             QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
             ui->textEdit_Infor->append(QString("实时相机数据采集:%1，Height:%2,Width:%3,FPS:%4,Count:%5").arg(formattedTime).arg(height).arg(width).arg(fps).arg(count));
          }

          std::vector<cv::Point2f> featurePoints;
          double qualityLevel = 0.01;
          int minDistance = 10;
          int blockSize = 3;
          bool useHarrisDetector = false;
          double k = 0.04;
          int maxCorners = 5000;
          cv::Mat frame, gray;
          std::vector<cv::Point2f> pts[2];
          std::vector<cv::Point2f> initPoints;
          std::vector<uchar> status;
          std::vector<float> err;
          cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
          double derivlambda = 0.5;
          int flags = 0;

          // detect first frame and find corners in it
          cv::Mat old_frame, old_gray;
          capture.read(old_frame);
          cvtColor(old_frame, old_gray, cv::COLOR_BGR2GRAY);
          goodFeaturesToTrack(old_gray, featurePoints, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);
          initPoints.insert(initPoints.end(), featurePoints.begin(), featurePoints.end());
          pts[0].insert(pts[0].end(), featurePoints.begin(), featurePoints.end());
          int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
          int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
          cv::Mat result = cv::Mat::zeros(cv::Size(width * 2, height), CV_8UC3);
          cv::Rect roi(0, 0, width, height);
          while (BasicImageGlobel::shouldStop) {
              bool ret = capture.read(frame);
              if (!ret) break;
              imshow("frame", frame);
              roi.x = 0;
              frame.copyTo(result(roi));
              cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

              // calculate optical flow
              calcOpticalFlowPyrLK(old_gray, gray, pts[0], pts[1], status, err, cv::Size(31, 31), 3, criteria, derivlambda, flags);
              size_t i, k;
              for (i = k = 0; i < pts[1].size(); i++)
              {
                  // 距离与状态测量
                  double dist = abs(pts[0][i].x - pts[1][i].x) + abs(pts[0][i].y - pts[1][i].y);
                  if (status[i] && dist > 2) {
                      pts[0][k] = pts[0][i];
                      initPoints[k] = initPoints[i];
                      pts[1][k++] = pts[1][i];
                      circle(frame, pts[1][i], 3, cv::Scalar(0, 255, 0), -1, 8);
                  }
              }
              // resize 有用特征点
              pts[1].resize(k);
              pts[0].resize(k);
              initPoints.resize(k);
              // 绘制跟踪轨迹
              draw_lines(frame, initPoints, pts[1]);
              roi.x = width;
              frame.copyTo(result(roi));
              cv::waitKey(50);
              // update old
              std::swap(pts[1], pts[0]);
              cv::swap(old_gray, gray);

              // need to re-init
              if (initPoints.size() < 40) {
                  goodFeaturesToTrack(old_gray, featurePoints, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);
                  initPoints.insert(initPoints.end(), featurePoints.begin(), featurePoints.end());
                  pts[0].insert(pts[0].end(), featurePoints.begin(), featurePoints.end());
                  printf("total feature points : %d\n", pts[0].size());
              }
          }
          //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
          QPixmap pix = matToQPixmap(frame);
          pix.scaled(ui->Lab_Image->size(),Qt::KeepAspectRatio);
          ui->Lab_Image->setScaledContents(true);
          ui->Lab_Image->setPixmap(pix);

        }else if(state==Qt::Unchecked){
            BasicImageGlobel::shouldStop=0;
            capture.release();//释放相机数据
        }
}
//稠密光流
void ImbaProcess::on_checkBox_DenseOpticalFlow_stateChanged(int state){
    cv::VideoCapture capture;
    //基于颜色对象跟踪选中计算
    if(state==Qt::Checked){
        BasicImageGlobel::shouldStop=1;
          if(ui->checkBox_VideoAnalysLocalfile->checkState()==Qt::Checked){//选择加载本地视频
              QString lastVideoPath = QFileDialog::getOpenFileName(this, "请选择视频", QStandardPaths::writableLocation(QStandardPaths::MoviesLocation), "视频(*.avi *.mp4);;");
              if(lastVideoPath.isEmpty())
              {
                  //停止数据采集，向信息区中写入信息
                  QDateTime currentDateTime = QDateTime::currentDateTime();
                  QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                  ui->textEdit_Infor->append(QString("本地视频数据采集:%1，视频背景分析失败，路径为空").arg(formattedTime));
                  return;
              }
             bool ret1 =capture.open(lastVideoPath.toStdString());
             int height =capture.get(cv::CAP_PROP_FRAME_HEIGHT);//视频高
             int width  =capture.get(cv::CAP_PROP_FRAME_WIDTH);//视频宽
             double fps =capture.get(cv::CAP_PROP_FPS);//视频帧率
             double count =capture.get(cv::CAP_PROP_FRAME_COUNT);//总帧率
             //在信息区显示当前加载视频的信息格式
             QDateTime currentDateTime = QDateTime::currentDateTime();
             QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
             ui->textEdit_Infor->append(QString("实时相机数据采集:%1，Height:%2,Width:%3,FPS:%4,Count:%5").arg(formattedTime).arg(height).arg(width).arg(fps).arg(count));
          }else if(ui->checkBox_VideoAnalysCameraData->checkState()==Qt::Checked){//选择加载相机数据
             bool ret2 =capture.open(ui->spinBox_VideoAnalysCameraNum->value());
             int height =capture.get(cv::CAP_PROP_FRAME_HEIGHT);//视频高
             int width  =capture.get(cv::CAP_PROP_FRAME_WIDTH);//视频宽
             double fps =capture.get(cv::CAP_PROP_FPS);//视频帧率
             double count =capture.get(cv::CAP_PROP_FRAME_COUNT);//总帧率
             //在信息区显示当前加载视频的信息格式
             QDateTime currentDateTime = QDateTime::currentDateTime();
             QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
             ui->textEdit_Infor->append(QString("实时相机数据采集:%1，Height:%2,Width:%3,FPS:%4,Count:%5").arg(formattedTime).arg(height).arg(width).arg(fps).arg(count));
          }
          cv::Mat preFrame, preGray;
          capture.read(preFrame);
          cvtColor(preFrame, preGray, cv::COLOR_BGR2GRAY);
          cv::Mat hsv = cv::Mat::zeros(preFrame.size(), preFrame.type());
          cv::Mat frame, gray;
          cv::Mat_<cv::Point2f> flow;
          std::vector<cv::Mat> mv;
          split(hsv, mv);
          cv::Mat mag =  cv::Mat::zeros(hsv.size(), CV_32FC1);
          cv::Mat ang =  cv::Mat::zeros(hsv.size(), CV_32FC1);
          cv::Mat xpts = cv::Mat::zeros(hsv.size(), CV_32FC1);
          cv::Mat ypts = cv::Mat::zeros(hsv.size(), CV_32FC1);
          while (BasicImageGlobel::shouldStop) {
              int64 start = cv::getTickCount();
              bool ret = capture.read(frame);
              if (!ret) break;
              imshow("frame", frame);
              cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
              calcOpticalFlowFarneback(preGray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
              for (int row = 0; row < flow.rows; row++)
              {
                  for (int col = 0; col < flow.cols; col++)
                  {
                      const cv::Point2f& flow_xy = flow.at<cv::Point2f>(row, col);
                      xpts.at<float>(row, col) = flow_xy.x;
                      ypts.at<float>(row, col) = flow_xy.y;
                  }
              }
              cartToPolar(xpts, ypts, mag, ang);
              ang = ang * 180.0 / CV_PI / 2.0;
              normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
              convertScaleAbs(mag, mag);
              convertScaleAbs(ang, ang);
              mv[0] = ang;
              mv[1] = cv::Scalar(255);
              mv[2] = mag;
              merge(mv, hsv);
              cv::Mat bgr;
              cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
              double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
              putText(bgr, cv::format("FPS : %.2f", fps), cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8);
              imshow("result", bgr);
              cv::waitKey(1);
          }
          //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
          QPixmap pix = matToQPixmap(frame);
          pix.scaled(ui->Lab_Image->size(),Qt::KeepAspectRatio);
          ui->Lab_Image->setScaledContents(true);
          ui->Lab_Image->setPixmap(pix);
        }else if(state==Qt::Unchecked){
            BasicImageGlobel::shouldStop=0;
            capture.release();//释放相机数据
        }
}
//均值迁移
void ImbaProcess::on_checkBox_MeanTransfer_stateChanged(int state){
    cv::VideoCapture capture;
    //基于颜色对象跟踪选中计算
    if(state==Qt::Checked){
        BasicImageGlobel::shouldStop=1;
          if(ui->checkBox_VideoAnalysLocalfile->checkState()==Qt::Checked){//选择加载本地视频
              QString lastVideoPath = QFileDialog::getOpenFileName(this, "请选择视频", QStandardPaths::writableLocation(QStandardPaths::MoviesLocation), "视频(*.avi *.mp4);;");
              if(lastVideoPath.isEmpty())
              {
                  //停止数据采集，向信息区中写入信息
                  QDateTime currentDateTime = QDateTime::currentDateTime();
                  QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
                  ui->textEdit_Infor->append(QString("本地视频数据采集:%1，视频背景分析失败，路径为空").arg(formattedTime));
                  return;
              }
             bool ret1 =capture.open(lastVideoPath.toStdString());
             int height =capture.get(cv::CAP_PROP_FRAME_HEIGHT);//视频高
             int width  =capture.get(cv::CAP_PROP_FRAME_WIDTH);//视频宽
             double fps =capture.get(cv::CAP_PROP_FPS);//视频帧率
             double count =capture.get(cv::CAP_PROP_FRAME_COUNT);//总帧率
             //在信息区显示当前加载视频的信息格式
             QDateTime currentDateTime = QDateTime::currentDateTime();
             QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
             ui->textEdit_Infor->append(QString("实时相机数据采集:%1，Height:%2,Width:%3,FPS:%4,Count:%5").arg(formattedTime).arg(height).arg(width).arg(fps).arg(count));
          }else if(ui->checkBox_VideoAnalysCameraData->checkState()==Qt::Checked){//选择加载相机数据
             bool ret2 =capture.open(ui->spinBox_VideoAnalysCameraNum->value());
             int height =capture.get(cv::CAP_PROP_FRAME_HEIGHT);//视频高
             int width  =capture.get(cv::CAP_PROP_FRAME_WIDTH);//视频宽
             double fps =capture.get(cv::CAP_PROP_FPS);//视频帧率
             double count =capture.get(cv::CAP_PROP_FRAME_COUNT);//总帧率
             //在信息区显示当前加载视频的信息格式
             QDateTime currentDateTime = QDateTime::currentDateTime();
             QString formattedTime =currentDateTime.toString("yyyy-MM-dd hh:mm:ss");
             ui->textEdit_Infor->append(QString("实时相机数据采集:%1，Height:%2,Width:%3,FPS:%4,Count:%5").arg(formattedTime).arg(height).arg(width).arg(fps).arg(count));
          }
          cv::Mat image;
          bool selectObject = false;
          int trackObject = 0;
          cv::Point origin;
          cv::Rect trackWindow;
          int hsize = 16;
          float hranges[] = { 0,180 };
          const float* phranges = hranges;
          namedWindow("MeanShift Demo", cv::WINDOW_AUTOSIZE);

          cv::Mat frame, hsv, hue, mask, hist, backproj;
          bool paused = false;
          capture.read(frame);
          cv::Rect selection = selectROI("MeanShift Demo", frame, true, false);

          while (BasicImageGlobel::shouldStop)
          {
              bool ret = capture.read(frame);
              if (!ret) break;
              frame.copyTo(image);
              cvtColor(image, hsv, cv::COLOR_BGR2HSV);
              inRange(hsv, cv::Scalar(26, 43, 46), cv::Scalar(34, 255, 255), mask);
              int ch[] = { 0, 0 };
              hue.create(hsv.size(), hsv.depth());
              mixChannels(&hsv, 1, &hue, 1, ch, 1);

              if (trackObject <= 0)
              {
                  // 建立搜索窗口与ROI区域直方图信息
                  cv::Mat roi(hue, selection), maskroi(mask, selection);
                  calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                  normalize(hist, hist, 0, 255, cv::NORM_MINMAX);

                  trackWindow = selection;
                  trackObject = 1;
              }
              // 反向投影
              calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
              backproj &= mask;
              // 均值迁移
              cv::meanShift(backproj, trackWindow, cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1));
              cv::rectangle(image, trackWindow, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
          }
             //将Mat对象转换为QpixMap对象,然后将图像大小设置成Qlabel大小并赋值到Qlabel中去
             QPixmap pix = matToQPixmap(image);
             pix.scaled(ui->Lab_Image->size(),Qt::KeepAspectRatio);
             ui->Lab_Image->setScaledContents(true);
             ui->Lab_Image->setPixmap(pix);
             cv::waitKey(10);
             ui->Lab_Image->clear();
             capture.release();
        }else if(state==Qt::Unchecked){
            BasicImageGlobel::shouldStop=0;
            capture.release();//释放相机数据
        }
}











