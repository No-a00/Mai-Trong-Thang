#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <windows.h>

using namespace cv;
using namespace std;

// Hàm mở hộp thoại chọn ảnh
string openFileDialog()
{
    char filename[MAX_PATH] = "";
    OPENFILENAMEA ofn;
    ZeroMemory(&ofn, sizeof(ofn));

    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = NULL;
    ofn.lpstrFilter = "Image Files\0*.jpg;*.jpeg;*.png;*.bmp;*.tiff\0All Files\0*.*\0";
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
    ofn.lpstrTitle = "Select an Image";

    if (GetOpenFileNameA(&ofn))
        return string(filename);
    else
        return "";
}

int main()
{
    
    //  Chọn và đọc ảnh 
    string path = openFileDialog();
    if (path.empty())
    {
        cout << "No image selected!" << endl;
        return 0;
    }
    Mat imgOriginal = imread(path, IMREAD_COLOR);
    Mat img = imread(path, IMREAD_GRAYSCALE);
    if (img.empty())
    {
        cout << "Failed to load image!" << endl;
        return -1;
    }

    cout << "Selected: " << path << endl;

    //  Áp dụng các bộ lọc 
    Mat sobelX, sobelY, sobel, prewittX, prewittY, prewitt, laplacian;
    Sobel(img, sobelX, CV_64F, 1, 0, 3);
    Sobel(img, sobelY, CV_64F, 0, 1, 3);
    magnitude(sobelX, sobelY, sobel);

    // Kernel Prewitt
    Mat kx = (Mat_<double>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
    Mat ky = (Mat_<double>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1);
    filter2D(img, prewittX, CV_64F, kx);
    filter2D(img, prewittY, CV_64F, ky);
    magnitude(prewittX, prewittY, prewitt);

    Laplacian(img, laplacian, CV_64F);

    //   Thêm nhiễu Gaussian 
    Mat noise(img.size(), CV_64F);
    randn(noise, 0, 10); 
    Mat noisy = Mat_<double>(img) + noise;
    noisy.convertTo(noisy, CV_8U);

    //   Sobel trên ảnh nhiễu 
    Mat sobelXn, sobelYn, sobelNoisy;
    Sobel(noisy, sobelXn, CV_64F, 1, 0, 3);
    Sobel(noisy, sobelYn, CV_64F, 0, 1, 3);
    magnitude(sobelXn, sobelYn, sobelNoisy);

    //   Lọc trong miền tần số (DFT) 
    Mat imgFloat;
    noisy.convertTo(imgFloat, CV_32F);
    Mat dftMat;
    dft(imgFloat, dftMat, DFT_COMPLEX_OUTPUT);

    // Bộ lọc thông thấp tròn
    Mat mask(img.size(), CV_32FC2, Scalar(0, 0));
    int r = 30;
    Point center(mask.cols / 2, mask.rows / 2);
    circle(mask, center, r, Scalar(1, 1), -1);
    Mat filtered;
    mulSpectrums(dftMat, mask, filtered, 0);

    Mat idftMat;
    idft(filtered, idftMat, DFT_SCALE | DFT_REAL_OUTPUT);
    idftMat.convertTo(idftMat, CV_8U);

    // Hiển thị kết quả 
    auto showSmall = [&](const string &name, const Mat &src)
    {
        Mat small;
        double scale = 500.0 / std::max(src.cols, src.rows);
        resize(src, small, Size(), scale, scale);
        imshow(name, small);
    };

    showSmall("Original Image", imgOriginal);
    showSmall("Sobel", abs(sobel) / 255);
    showSmall("Prewitt", abs(prewitt) / 255);
    showSmall("Laplacian", abs(laplacian) / 255);
    showSmall("Gaussian Noise", noisy);
    showSmall("Sobel (Noisy)", abs(sobelNoisy) / 255);
    showSmall("FFT Filtered Image", idftMat);

    waitKey(0);
    return 0;
}
