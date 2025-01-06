#include <iostream>
#include "tinyxml2.h"

using namespace tinyxml2;

int main() {
    XMLDocument doc;

    // Create the root element
    XMLElement* root = doc.NewElement("library");
    doc.InsertFirstChild(root);

    // Create book elements
    XMLElement* book1 = doc.NewElement("book");
    book1->SetAttribute("id", 1);
    book1->SetText("The Great Gatsby");
    root->InsertEndChild(book1);

    XMLElement* book2 = doc.NewElement("book");
    book2->SetAttribute("id", 2);
    book2->SetText("1984");
    root->InsertEndChild(book2);

    // Save the XML file
    XMLError eResult = doc.SaveFile("library.xml");
    if (eResult != XML_SUCCESS) {
        std::cerr << "Error saving XML file: " << doc.ErrorIDToName(eResult) << std::endl;
        return eResult;
    }

    std::cout << "XML file saved successfully." << std::endl;
    return 0;
}