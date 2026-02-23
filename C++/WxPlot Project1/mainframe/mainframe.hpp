#pragma once
#ifndef MAINFRAME_HPP
#define MAINFRAME_HPP

#include <wx/wx.h>
#include "../../wxplot.hpp"

/**
 * @class MainFrame
 * @brief This class represents the main example window for the application.
 *
 * The `MainFrame` class is a wxWidgets frame that acts as the main user interface
 * for the application. It contains wxPlot, and provides event handlers
 * for various events such as menu actions and window resizing.
 *
 * This frame is just an example responsible for setting up the UI, handling user interaction,
 * and managing the overall plot display.
 */
class MainFrame : public wxFrame
{
public:
    MainFrame();

private:
    void OnGenerateRandom(wxCommandEvent& event);
    void OnSize(wxSizeEvent& event);
    void GenerateRandomData();

    wxPlot* plot;
    wxButton* randomButton;  // This is correctly declared here
};

#endif // MAINFRAME_HPP