import wx
import xml.etree.ElementTree as ET
import os

class RecipeManager(wx.Frame):
    def __init__(self, *args, **kw):
        super(RecipeManager, self).__init__(*args, **kw)

        self.recipes = []
        self.InitUI()
        self.LoadData()

    def InitUI(self):
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Recipe Name
        vbox.Add(wx.StaticText(panel, label="Recipe Name:"), flag=wx.ALL, border=5)
        self.recipe_name_input = wx.TextCtrl(panel)
        vbox.Add(self.recipe_name_input, flag=wx.EXPAND | wx.ALL, border=5)

        # Ingredients
        vbox.Add(wx.StaticText(panel, label="Ingredients (one per line):"), flag=wx.ALL, border=5)
        self.ingredients_input = wx.TextCtrl(panel, style=wx.TE_MULTILINE)
        vbox.Add(self.ingredients_input, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

        # Instructions
        vbox.Add(wx.StaticText(panel, label="Cooking Instructions:"), flag=wx.ALL, border=5)
        self.instructions_input = wx.TextCtrl(panel, style=wx.TE_MULTILINE)
        vbox.Add(self.instructions_input, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

        # Buttons
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        add_btn = wx.Button(panel, label='Add Recipe')
        add_btn.Bind(wx.EVT_BUTTON, self.OnAddRecipe)
        hbox.Add(add_btn, flag=wx.RIGHT, border=5)

        edit_btn = wx.Button(panel, label='Edit Recipe')
        edit_btn.Bind(wx.EVT_BUTTON, self.OnEditRecipe)
        hbox.Add(edit_btn, flag=wx.RIGHT, border=5)

        delete_btn = wx.Button(panel, label='Delete Recipe')
        delete_btn.Bind(wx.EVT_BUTTON, self.OnDeleteRecipe)
        hbox.Add(delete_btn)

        vbox.Add(hbox, flag=wx.ALIGN_CENTER | wx.ALL, border=5)

        # Recipe List
        self.recipe_listbox = wx.ListBox(panel)
        self.recipe_listbox.Bind(wx.EVT_LISTBOX, self.OnSelectRecipe)
        vbox.Add(self.recipe_listbox, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

        panel.SetSizer(vbox)

        self.SetTitle('Recipe Manager')
        self.SetSize((400, 600))
        self.Centre()

    def OnAddRecipe(self, event):
        name = self.recipe_name_input.GetValue()
        ingredients = self.ingredients_input.GetValue().strip().splitlines()
        instructions = self.instructions_input.GetValue()

        if name and ingredients and instructions:
            recipe = {
                'name': name,
                'ingredients': ingredients,
                'instructions': instructions
            }
            self.recipes.append(recipe)
            self.UpdateRecipeList()
            self.SaveData()
            self.ClearInputs()

    def OnEditRecipe(self, event):
        selected_index = self.recipe_listbox.GetSelection()
        if selected_index != wx.NOT_FOUND:
            name = self.recipe_name_input.GetValue()
            ingredients = self.ingredients_input.GetValue().strip().splitlines()
            instructions = self.instructions_input.GetValue()

            if name and ingredients and instructions:
                self.recipes[selected_index] = {
                    'name': name,
                    'ingredients': ingredients,
                    'instructions': instructions
                }
                self.UpdateRecipeList()
                self.SaveData()
                self.ClearInputs()

    def OnDeleteRecipe(self, event):
        selected_index = self.recipe_listbox.GetSelection()
        if selected_index != wx.NOT_FOUND:
            del self.recipes[selected_index]
            self.UpdateRecipeList()
            self.SaveData()
            self.ClearInputs()

    def OnSelectRecipe(self, event):
        selected_index = self.recipe_listbox.GetSelection()
        if selected_index != wx.NOT_FOUND:
            recipe = self.recipes[selected_index]
            self.recipe_name_input.SetValue(recipe['name'])
            self.ingredients_input.SetValue('\n'.join(recipe['ingredients']))
            self.instructions_input.SetValue(recipe['instructions'])

    def UpdateRecipeList(self):
        self.recipe_listbox .Clear()
        for recipe in self.recipes:
            self.recipe_listbox.Append(recipe['name'])

    def ClearInputs(self):
        self.recipe_name_input.Clear()
        self.ingredients_input.Clear()
        self.instructions_input.Clear()

    def SaveData(self):
        root = ET.Element("Recipes")
        for recipe in self.recipes:
            recipe_elem = ET.SubElement(root, "Recipe")
            name_elem = ET.SubElement(recipe_elem, "Name")
            name_elem.text = recipe['name']
            ingredients_elem = ET.SubElement(recipe_elem, "Ingredients")
            for ingredient in recipe['ingredients']:
                ing_elem = ET.SubElement(ingredients_elem, "Ingredient")
                ing_elem.text = ingredient
            instructions_elem = ET.SubElement(recipe_elem, "Instructions")
            instructions_elem.text = recipe['instructions']

        tree = ET.ElementTree(root)
        with open("recipes.xml", "wb") as fh:
            tree.write(fh)

    def LoadData(self):
        if os.path.exists("recipes.xml"):
            tree = ET.parse("recipes.xml")
            root = tree.getroot()
            for recipe_elem in root.findall("Recipe"):
                name = recipe_elem.find("Name").text
                ingredients = [ing.text for ing in recipe_elem.find("Ingredients").findall("Ingredient")]
                instructions = recipe_elem.find("Instructions").text
                self.recipes.append({
                    'name': name,
                    'ingredients': ingredients,
                    'instructions': instructions
                })
            self.UpdateRecipeList()

if __name__ == '__main__':
    app = wx.App()
    frame = RecipeManager(None)
    frame.Show()
    app.MainLoop()