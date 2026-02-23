#include "mainframe.hpp"
#include <random>
#include <ctime>

MainFrame::MainFrame() : wxFrame(nullptr, wxID_ANY, "Random Plot", wxDefaultPosition, wxSize(800, 600))
{
    // Create status bar FIRST - before anything else that might try to use it
    CreateStatusBar();

    // Set black background for the frame
    SetBackgroundColour(wxColour(0, 0, 0));

    // Create a panel with black background
    wxPanel* mainPanel = new wxPanel(this, wxID_ANY);
    mainPanel->SetBackgroundColour(wxColour(0, 0, 0));

    // Create vertical sizer for layout
    wxBoxSizer* mainSizer = new wxBoxSizer(wxVERTICAL);

    // Create the plot
    plot = new wxPlot(mainPanel, WXPLOT_FIGURE_2D, WXPLOT_TYPE_LINE_SCATTER);

    // Configure plot for dark theme
    plot->SetBackgroundColour(wxColour(0, 0, 0));
    plot->setFontSize(12);
    plot->setTitle("Random Data Generator");
    plot->setYlabel("Value");
    plot->setXlabel("Sample Index");
    plot->setTicks(5, 5);
    plot->gridOn(true);
    plot->legendOn(false);
    plot->setRadius(4);
    plot->fillCircles(true);

    // Create the button with dark styling
    randomButton = new wxButton(mainPanel, wxID_ANY, "Generate Random Data");
    randomButton->SetBackgroundColour(wxColour(60, 60, 60));
    randomButton->SetForegroundColour(wxColour(255, 255, 255));
    randomButton->SetFont(wxFont(12, wxFONTFAMILY_DEFAULT, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD));

    // Add widgets to sizer
    mainSizer->Add(plot, 1, wxEXPAND | wxALL, 10);
    mainSizer->Add(randomButton, 0, wxALIGN_CENTER | wxALL, 15);

    mainPanel->SetSizer(mainSizer);

    // Generate initial random data
    GenerateRandomData();

    // Set initial plot dimensions
    wxCoord width, height;
    GetClientSize(&width, &height);
    plot->setPlotStartWidth(70);
    plot->setPlotStartHeight(70);
    plot->setPlotEndWidth(width - 70);
    plot->setPlotEndHeight(height - 100);

    // Bind events
    randomButton->Bind(wxEVT_BUTTON, &MainFrame::OnGenerateRandom, this);
    Bind(wxEVT_SIZE, &MainFrame::OnSize, this);

    // NOW set the status bar text (after everything is initialized)
    SetStatusText("Click the button to generate new random data");
}
void MainFrame::GenerateRandomData()
{
    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-50.0, 50.0);

    // Generate data points
    std::vector<double> xData;
    std::vector<double> yData;

    int numPoints = 30;  // Fewer points for cleaner look

    for (int i = 0; i < numPoints; i++) {
        xData.push_back(i * 10);  // X: 0, 10, 20, ...
        yData.push_back(dis(gen)); // Random Y between -50 and 50
    }

    // Prepare data for plotting
    std::vector<std::vector<double>> plotData = { xData, yData };

    // Set the data
    plot->setData(plotData);

    // Set Y-axis limits with some padding
    plot->setYlim(-60, 60);

    // Refresh the plot
    plot->Refresh();

    // Update status bar
    SetStatusText(wxString::Format("Generated %d random data points", numPoints));
}

void MainFrame::OnGenerateRandom(wxCommandEvent& event)
{
    GenerateRandomData();
}

void MainFrame::OnSize(wxSizeEvent& event)
{
    event.Skip();

    // Get new window size
    wxCoord width, height;
    GetClientSize(&width, &height);

    if (plot) {
        // Update plot dimensions with margins
        // Leave 70px on each side, and extra space at bottom for button
        plot->setPlotStartWidth(70);
        plot->setPlotStartHeight(70);
        plot->setPlotEndWidth(width - 70);
        plot->setPlotEndHeight(height - 120);  // Extra space for button

        plot->Refresh();
    }

    // Note: We don't need to reference randomButton here at all
    // The button is managed by the sizer, so it will resize automatically
}