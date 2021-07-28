#include "Surface_Detection_Form.h"

using namespace System;
using namespace System::Windows::Forms;

[STAThread]
void Main(array<String^>^args)
{
	Application::EnableVisualStyles();
	Application::SetCompatibleTextRenderingDefault(false);
	Surface_Detection_Form::Surface_Detection_Form form;
	Application::Run(%form);
}