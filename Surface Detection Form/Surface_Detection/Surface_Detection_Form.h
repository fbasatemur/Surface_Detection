#pragma once
#include <iostream>
#include <atlstr.h>	

#include "KernelCpu.h"
#include "Conv2D.h"
#include "MaxPooling2D.h"
#include "Flatten.h"
#include "Dense.h"
#include "BatchNormalization.h"
#include "CpuMat.h"
#include "Image.h"

namespace Surface_Detection_Form {

	using namespace System;
	using namespace System::ComponentModel;
	using namespace System::Collections;
	using namespace System::Windows::Forms;
	using namespace System::Data;
	using namespace System::Drawing;

	/// <summary>
	/// Summary for Surface_Detection_Form
	/// </summary>
	public ref class Surface_Detection_Form : public System::Windows::Forms::Form
	{
	public:
		Surface_Detection_Form(void)
		{
			InitializeComponent();
			//
			//TODO: Add the constructor code here
			//

			acceptable = gcnew Bitmap("..\\database\\done.bmp");
			unacceptable = gcnew Bitmap("..\\database\\reject.bmp");

			// model definition
			//inputImage = new CpuMat(512, 512, 1, false);		// only with 10X intensity weights
			inputImage = new CpuMat(512, 512, 3, false);

			surface = gcnew Bitmap(inputImage->Cols, inputImage->Rows);
			pictureBox1->Image = surface;

			conv = new Conv2D(8, 3, 3, inputImage);
			maxPool = new MaxPooling2D(conv->Result, 2, 2, 2, 2);
			conv1 = new Conv2D(16, 3, 3, maxPool->Result);
			maxPool1 = new MaxPooling2D(conv1->Result, 2, 2, 2, 2);
			flatten = new Flatten(maxPool1->Result);
			dense = new Dense(16, flatten->Result->Rows, flatten->Result->Cols);
			batchNorm = new BatchNormalization(dense->Result->Rows, dense->Result->Cols);
			dense1 = new Dense(1, dense->Result->Rows, dense->Result->Cols, false);

		}

	protected:
		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		~Surface_Detection_Form()
		{
			if (components)
			{
				delete components;
			}
		}
	private: System::Windows::Forms::PictureBox^ pictureBox1;
	private: System::Windows::Forms::MenuStrip^ menuStrip1;
	private: System::Windows::Forms::ToolStripMenuItem^ fileToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ openToolStripMenuItem;
	private: System::Windows::Forms::OpenFileDialog^ openFileDialog1;
	protected:

	private:
		/// <summary>
		/// Required designer variable.
		/// </summary>
		System::ComponentModel::Container^ components;
		float TH;						// there are different thresholds for both 10X and 40X images
		bool isWeightSelected = false;
		Bitmap^ surface;
		Bitmap^ acceptable;
		Bitmap^ unacceptable;
		CpuMat* inputImage;
		Conv2D* conv;
		Conv2D* conv1;
		MaxPooling2D* maxPool;
		MaxPooling2D* maxPool1;
		Flatten* flatten;
		Dense* dense;
		Dense* dense1;
		BatchNormalization* batchNorm;
	private: System::Windows::Forms::Label^ label1;
	private: System::Windows::Forms::Label^ label2;
	private: System::Windows::Forms::PictureBox^ pictureBox2;
	private: System::Windows::Forms::ToolStripMenuItem^ weightsToolStripMenuItem;
	private: System::Windows::Forms::ToolStripMenuItem^ xToolStripMenuItem10X;
	private: System::Windows::Forms::ToolStripMenuItem^ xToolStripMenuItem40X;


	private: System::Windows::Forms::Label^ imagePathLbl;


		   void InputProcessIntensity(float* input, int width, int height, float* result) {

			   int newWidth = int(width / 2);
			   int newHeight = int(height / 2);
			   int size2D = newWidth * newHeight;

			   int newRow = 0, newCol = 0;

			   for (int row = 0; row < height; row += 2) {
				   newCol = 0;
				   for (int col = 0; col < width; col += 2)
				   {
					   result[newRow * newWidth + newCol] = float(input[row * width + col]) / 255.0F;

					   surface->SetPixel(col, row, Color::FromArgb(BYTE(input[row * width + col]), BYTE(input[row * width + col]), BYTE(input[row * width + col])));

					   newCol++;
				   }
				   newRow++;
			   }
		   }

		   void InputProcessBGR(BYTE* input, int width, int height, float* result) {

			   int newWidth = int(width / 2);
			   int newHeight = int(height / 2);
			   int resultSize2D = newWidth * newHeight;

			   float* resultR, * resultG, * resultB;
			   int newRow = 0, newCol;
			   int psw, inputPos, resultPos;
			   psw = width * 3;

			   resultB = result;
			   resultG = result + resultSize2D;
			   resultR = result + 2 * resultSize2D;

			   for (int row = 0; row < height; row += 2) {
				   newCol = 0;
				   for (int col = 0; col < width; col += 2)
				   {
					   inputPos = (height - row - 1) * psw + col * 3;
					   resultPos = newRow * newWidth + newCol;
					   resultB[resultPos] = float(input[inputPos]) / 255.0F;
					   resultG[resultPos] = float(input[inputPos + 1]) / 255.0F;
					   resultR[resultPos] = float(input[inputPos + 2]) / 255.0F;

					   surface->SetPixel(newCol, newRow, Color::FromArgb(int(input[inputPos + 2]), int(input[inputPos + 1]), int(input[inputPos])));
					   newCol++;
				   }
				   newRow++;
			   }
		   }

		   void LoadWeights(std::string weightFolder) {

			   std::string conv2DKernel = weightFolder + "conv2d\\kernel.txt";
			   std::string conv2DBias = weightFolder + "conv2d\\bias.txt";

			   std::string conv2D1Kernel = weightFolder + "conv2d_1\\kernel.txt";
			   std::string conv2D1Bias = weightFolder + "conv2d_1\\bias.txt";

			   std::string denseKernel = weightFolder + "dense\\kernel.txt";
			   std::string denseBias = weightFolder + "dense\\bias.txt";

			   std::string dense1Kernel = weightFolder + "dense_1\\kernel.txt";
			   std::string dense1Bias = weightFolder + "dense_1\\bias.txt";

			   std::string batchNormBeta = weightFolder + "batch_normalization\\beta.txt";
			   std::string batchNormGamma = weightFolder + "batch_normalization\\gamma.txt";
			   std::string batchNormMovingMean = weightFolder + "batch_normalization\\moving_mean.txt";
			   std::string batchNormMovingVariance = weightFolder + "batch_normalization\\moving_variance.txt";

			   //////// load kernel and bias weights
			   conv->load(conv2DKernel, conv2DBias);
			   conv1->load(conv2D1Kernel, conv2D1Bias);

			   dense->load(denseKernel, denseBias);
			   dense1->load(dense1Kernel, dense1Bias);

			   // load batchnormalization layer weights
			   batchNorm->load(batchNormBeta, batchNormGamma, batchNormMovingMean, batchNormMovingVariance);
		   }

#pragma region Windows Form Designer generated code
		   /// <summary>
		   /// Required method for Designer support - do not modify
		   /// the contents of this method with the code editor.
		   /// </summary>
		   void InitializeComponent(void)
		   {
			   this->pictureBox1 = (gcnew System::Windows::Forms::PictureBox());
			   this->menuStrip1 = (gcnew System::Windows::Forms::MenuStrip());
			   this->fileToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			   this->openToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			   this->weightsToolStripMenuItem = (gcnew System::Windows::Forms::ToolStripMenuItem());
			   this->xToolStripMenuItem10X = (gcnew System::Windows::Forms::ToolStripMenuItem());
			   this->xToolStripMenuItem40X = (gcnew System::Windows::Forms::ToolStripMenuItem());
			   this->openFileDialog1 = (gcnew System::Windows::Forms::OpenFileDialog());
			   this->imagePathLbl = (gcnew System::Windows::Forms::Label());
			   this->label1 = (gcnew System::Windows::Forms::Label());
			   this->label2 = (gcnew System::Windows::Forms::Label());
			   this->pictureBox2 = (gcnew System::Windows::Forms::PictureBox());
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->BeginInit();
			   this->menuStrip1->SuspendLayout();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox2))->BeginInit();
			   this->SuspendLayout();
			   // 
			   // pictureBox1
			   // 
			   this->pictureBox1->Location = System::Drawing::Point(16, 66);
			   this->pictureBox1->Name = L"pictureBox1";
			   this->pictureBox1->Size = System::Drawing::Size(600, 600);
			   this->pictureBox1->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			   this->pictureBox1->TabIndex = 0;
			   this->pictureBox1->TabStop = false;
			   // 
			   // menuStrip1
			   // 
			   this->menuStrip1->BackColor = System::Drawing::SystemColors::ControlLightLight;
			   this->menuStrip1->ImageScalingSize = System::Drawing::Size(20, 20);
			   this->menuStrip1->Items->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(1) { this->fileToolStripMenuItem });
			   this->menuStrip1->Location = System::Drawing::Point(0, 0);
			   this->menuStrip1->Name = L"menuStrip1";
			   this->menuStrip1->Size = System::Drawing::Size(1270, 28);
			   this->menuStrip1->TabIndex = 1;
			   this->menuStrip1->Text = L"menuStrip1";
			   // 
			   // fileToolStripMenuItem
			   // 
			   this->fileToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {
				   this->openToolStripMenuItem,
					   this->weightsToolStripMenuItem
			   });
			   this->fileToolStripMenuItem->Name = L"fileToolStripMenuItem";
			   this->fileToolStripMenuItem->Size = System::Drawing::Size(46, 24);
			   this->fileToolStripMenuItem->Text = L"File";
			   // 
			   // openToolStripMenuItem
			   // 
			   this->openToolStripMenuItem->Name = L"openToolStripMenuItem";
			   this->openToolStripMenuItem->Size = System::Drawing::Size(224, 26);
			   this->openToolStripMenuItem->Text = L"Open";
			   this->openToolStripMenuItem->Click += gcnew System::EventHandler(this, &Surface_Detection_Form::openToolStripMenuItem_Click);
			   // 
			   // weightsToolStripMenuItem
			   // 
			   this->weightsToolStripMenuItem->DropDownItems->AddRange(gcnew cli::array< System::Windows::Forms::ToolStripItem^  >(2) {
				   this->xToolStripMenuItem10X,
					   this->xToolStripMenuItem40X
			   });
			   this->weightsToolStripMenuItem->Name = L"weightsToolStripMenuItem";
			   this->weightsToolStripMenuItem->Size = System::Drawing::Size(224, 26);
			   this->weightsToolStripMenuItem->Text = L"Weights";
			   // 
			   // xToolStripMenuItem10X
			   // 
			   this->xToolStripMenuItem10X->Name = L"xToolStripMenuItem10X";
			   this->xToolStripMenuItem10X->Size = System::Drawing::Size(224, 26);
			   this->xToolStripMenuItem10X->Text = L"10X";
			   this->xToolStripMenuItem10X->Click += gcnew System::EventHandler(this, &Surface_Detection_Form::xToolStripMenuItem10X_Click);
			   // 
			   // xToolStripMenuItem40X
			   // 
			   this->xToolStripMenuItem40X->Name = L"xToolStripMenuItem40X";
			   this->xToolStripMenuItem40X->Size = System::Drawing::Size(224, 26);
			   this->xToolStripMenuItem40X->Text = L"40X";
			   this->xToolStripMenuItem40X->Click += gcnew System::EventHandler(this, &Surface_Detection_Form::xToolStripMenuItem40X_Click);
			   // 
			   // openFileDialog1
			   // 
			   this->openFileDialog1->FileName = L"openFileDialog1";
			   this->openFileDialog1->Filter = L"bmp files (*.bmp)|*.bmp";
			   this->openFileDialog1->Multiselect = true;
			   // 
			   // imagePathLbl
			   // 
			   this->imagePathLbl->AutoSize = true;
			   this->imagePathLbl->Location = System::Drawing::Point(13, 43);
			   this->imagePathLbl->Name = L"imagePathLbl";
			   this->imagePathLbl->Size = System::Drawing::Size(45, 17);
			   this->imagePathLbl->TabIndex = 2;
			   this->imagePathLbl->Text = L"Path: ";
			   // 
			   // label1
			   // 
			   this->label1->AutoSize = true;
			   this->label1->Location = System::Drawing::Point(702, 43);
			   this->label1->Name = L"label1";
			   this->label1->Size = System::Drawing::Size(20, 17);
			   this->label1->TabIndex = 3;
			   this->label1->Text = L"...";
			   // 
			   // label2
			   // 
			   this->label2->AutoSize = true;
			   this->label2->Location = System::Drawing::Point(636, 43);
			   this->label2->Name = L"label2";
			   this->label2->Size = System::Drawing::Size(60, 17);
			   this->label2->TabIndex = 4;
			   this->label2->Text = L"Predict: ";
			   // 
			   // pictureBox2
			   // 
			   this->pictureBox2->Location = System::Drawing::Point(639, 66);
			   this->pictureBox2->Name = L"pictureBox2";
			   this->pictureBox2->Size = System::Drawing::Size(600, 600);
			   this->pictureBox2->SizeMode = System::Windows::Forms::PictureBoxSizeMode::Zoom;
			   this->pictureBox2->TabIndex = 5;
			   this->pictureBox2->TabStop = false;
			   // 
			   // Surface_Detection_Form
			   // 
			   this->AutoScaleDimensions = System::Drawing::SizeF(8, 16);
			   this->AutoScaleMode = System::Windows::Forms::AutoScaleMode::Font;
			   this->BackColor = System::Drawing::SystemColors::ControlLightLight;
			   this->ClientSize = System::Drawing::Size(1270, 682);
			   this->Controls->Add(this->pictureBox2);
			   this->Controls->Add(this->label2);
			   this->Controls->Add(this->label1);
			   this->Controls->Add(this->imagePathLbl);
			   this->Controls->Add(this->pictureBox1);
			   this->Controls->Add(this->menuStrip1);
			   this->MainMenuStrip = this->menuStrip1;
			   this->Margin = System::Windows::Forms::Padding(4);
			   this->Name = L"Surface_Detection_Form";
			   this->Text = L"Surface Detection";
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox1))->EndInit();
			   this->menuStrip1->ResumeLayout(false);
			   this->menuStrip1->PerformLayout();
			   (cli::safe_cast<System::ComponentModel::ISupportInitialize^>(this->pictureBox2))->EndInit();
			   this->ResumeLayout(false);
			   this->PerformLayout();

		   }
#pragma endregion
	private: System::Void openToolStripMenuItem_Click(System::Object^ sender, System::EventArgs^ e) {

		if (!isWeightSelected) {
			MessageBox::Show("Please select a weight type !", "Select Weight");
		}

		else if (openFileDialog1->ShowDialog() == System::Windows::Forms::DialogResult::OK) {

			long size;
			int width, height;
			BYTE* buffer;
			float* intensity;
			float* predict;
			clock_t totalTime = 0;
			clock_t start;
			clock_t end;


			for (int imageIndex = 0; imageIndex < openFileDialog1->FileNames->GetLength(0); imageIndex++)
			{
				imagePathLbl->Text = openFileDialog1->FileNames[imageIndex];
				imagePathLbl->Refresh();
				CString str = openFileDialog1->FileNames[imageIndex];

				// read BMP image
				buffer = LoadBMP(width, height, size, (LPCTSTR)str);
				InputProcessBGR(buffer, width, height, (float*)inputImage->CpuP);

				// only with 10X intensity images
				/*intensity = ConvertBMPToIntensity(buffer, width, height);
				InputProcessIntensity(intensity, width, height, (float*)inputImage->CpuP);*/

				conv->apply(inputImage);
				cpuRelu(conv->Result);
				maxPool->apply(conv->Result);

				conv1->apply(maxPool->Result);
				cpuRelu(conv1->Result);
				maxPool1->apply(conv1->Result);

				flatten->apply(maxPool1->Result);

				dense->apply(flatten->Result);
				cpuRelu(dense->Result);
				batchNorm->apply(dense->Result);

				dense1->apply(dense->Result);
				cpuSigmoid(dense1->Result);

				predict = (float*)dense1->Result->CpuP;
				label1->Text = predict[0].ToString();
				pictureBox1->Refresh();
				label1->Refresh();

				if (predict[0] > TH)
					pictureBox2->Image = acceptable;
				else
					pictureBox2->Image = unacceptable;

				pictureBox2->Refresh();

				delete[] intensity;
				delete[] buffer;
			}
		}

	}
	private: System::Void xToolStripMenuItem10X_Click(System::Object^ sender, System::EventArgs^ e) {

		//LoadWeights("..\\database\\model_save_10X_intensity\\");		// only with 10X intensity images

		LoadWeights("..\\database\\model_save_10X_BGR\\");
		TH = 0.5F;
		MessageBox::Show("10X Weights loaded", "Info");
		isWeightSelected = true;
	}
	private: System::Void xToolStripMenuItem40X_Click(System::Object^ sender, System::EventArgs^ e) {

		LoadWeights("..\\database\\model_save_40X_BGR\\");
		TH = 0.05F;
		MessageBox::Show("40X Weights loaded", "Info");
		isWeightSelected = true;
	}
	};
}
