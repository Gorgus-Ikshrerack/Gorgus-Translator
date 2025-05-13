#
# main.py
#
__VERSION__ = "1.10"
print("Loading...")

import os, sys, re
import importlib.util
import platform
import subprocess

# hopefully fix some issues
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("Checking for dependencies...")
GIT_AVAILABLE = True
try:
    import git
except ModuleNotFoundError:
    GIT_AVAILABLE = False

def install_module(module_name):
    if module_name == "gitpython":
        module_name = "git"

    if importlib.util.find_spec(module_name) is None:
        print(f"{module_name} not found. Installing...")
        
        # Determine correct pip command
        pip_cmd = "pip" if platform.system() == "Windows" else "pip3"
        
        try:
            if module_name == "git":
                module_name = "gitpython"

            subprocess.check_call([pip_cmd, "install", module_name])
            print(f"{module_name} installed successfully!")
        except subprocess.CalledProcessError:
            print(f"Error: Failed to install {module_name}. Please install it manually.")
            sys.exit(1)
    else:
        pass
        #print(f"{module_name} is already installed.")

with open("requirements.txt", "r") as f:
    for line in f.readlines():
        install_module(line.strip("\n"))

from rich import print as rich_print

rich_print("\n[bold]=== Gorgus Translator ===[/bold]")
rich_print(f"[bold]Version:[/bold] [dim cyan]{__VERSION__}[/dim cyan]\n")

if not GIT_AVAILABLE:
    rich_print("[bold bright_yellow]WARNING[/bold bright_yellow] [bold]gitpython[/bold] was not found! You will not be able to install automatic updates.")
else:
    rich_print("[bold bright_green]INFO[/bold bright_green] [bold]gitpython[/bold] found! :)")

rich_print("[bold bright_green]INFO[/bold bright_green] Loading [bold]textual[/bold]..")

from textual.app import App, ComposeResult, SystemCommand
from textual.widgets import TextArea, Header, Footer, TabbedContent, TabPane, Select, Label, MarkdownViewer, DataTable, Input, Rule, Checkbox, Button, Markdown
from textual.containers import Horizontal, Vertical, VerticalScroll, ItemGrid, Center
from textual import on, work, log, events
from textual.css.query import NoMatches
from textual.worker import WorkerState
from pyperclip import copy
from time import sleep

rich_print("[bold bright_green]INFO[/bold bright_green] Loading [bold]utility[/bold] functions..")
from util import get_settings, modify_json

rich_print("[bold bright_green]INFO[/bold bright_green] Loading [bold]games[/bold]..")
from widgets.game import Game, GameInfo
from games.wordle import WordleGame
from games.hangman import Hangman
from games.typing_game import TypingGame
from widgets.message_box import MessageBox

rich_print("[bold bright_green]INFO[/bold bright_green] Loading translation dictionary..")

from translations import translation_dictionary, phrase_translations, dictionary_information

rich_print("[bold bright_green]INFO[/bold bright_green] Starting translater..")

from translater import translate, get_ipa_pronounciation

rich_print("\n[bold bright_green]Done![/bold bright_green] Loading complete!")


GAMES_MD = """\
# Games

Hi! Currently there aren't very many Gorgus Games™ here yet, so uhh...  
Enjoy off brand Wordle and Hangman lmao. 

Oh yeah, and there's a multiplayer game too. :P
Both people need the translator though.
"""

GAMES = [
    GameInfo(
        "Gordle",
        "Each day a new random word in the translator's Gorgus dictionary is chosen, and you have to guess it within 6 tries! (this is just Wordle)",
        WordleGame
    ),
    GameInfo(
        "Bingbonk norack",
        "it's just hangman but with gorgus words bro",
        Hangman
    ),
    GameInfo(
        "Yutik spek",
        "You and another person race to see who can define a word faster!!! :)",
        TypingGame
    )
]



class CopyableLabel(Label):
    def __init__(self, text = "", copy_msg = "Text copied to clipboard!", **kwargs):
        super().__init__(text, **kwargs)
        self.text = re.sub(r'\[.*?\]', '', text)
        self.copy_msg = copy_msg

    def update(self, content = ""):
        self.text = re.sub(r'\[.*?\]', '', content)
        return super().update(content)

    @on(events.Click)
    def copy_stuff(self, _) -> None:
        """Copies the text to clipboard when clicked."""
        copy(self.text)
        self.notify(self.copy_msg, severity="info")

class GorgusTranslator(App):
    TITLE = "Gorgus Translator"
    #SUB_TITLE = "Made with ❤️ by @spookydervish"

    CSS_PATH = "resources/style.tcss"

    #ENABLE_COMMAND_PALETTE = False

    translation = ""
    translation_input = "Hey! How are you?"

    def get_system_commands(self, screen):
        yield SystemCommand(
            "Quit the application",
            "Quit the application as soon as possible",
            self.action_quit,
        )

        if screen.query("HelpPanel"):
            yield SystemCommand(
                "Hide keys and help panel",
                "Hide the keys and widget help panel",
                self.action_hide_help_panel,
            )
        else:
            yield SystemCommand(
                "Show keys and help panel",
                "Show help for the focused widget and a summary of available keys",
                self.action_show_help_panel,
            )

        if screen.maximized is not None:
            yield SystemCommand(
                "Minimize",
                "Minimize the widget and restore to normal size",
                screen.action_minimize,
            )
        elif screen.focused is not None and screen.focused.allow_maximize:
            yield SystemCommand(
                "Maximize", "Maximize the focused widget", screen.action_maximize
            )

        yield SystemCommand(
            "Save screenshot",
            "Save an SVG 'screenshot' of the current screen",
            self.deliver_screenshot,
        )

    def get_git_info(self):
        if not GIT_AVAILABLE:
            return "Gitpython is not installed, cannot get branch and version number"

        try:
            repo = git.Repo(search_parent_directories=True)
            branch = repo.active_branch.name
            version = repo.git.describe(tags=True, always=True)
            return branch, version
        except Exception:
            return "An error occured while getting the branch and version number. Did you [bold]git clone[/bold] the git repo?"  # Not a git repo or an error occurred

    @work(thread=True, group="updates", name="check-updates", exit_on_error=False)
    def check_for_updates(self):
        """This function will return `True` when updates are available, otherwise it will return `False`.
        """

        # disable the check for updates button
        self.query_one("#check-update-button").disabled = True
        self.query_one("#update-button").disabled = True

        version_label = self.query_one("#version-label")
        version_label.update(
            f"Branch: {self.git_info[0]} | Version: {self.git_info[1]} | Checking for updates..."
        )
        version_label.classes = "warning"

        self.log("Checking for updates..")

        self.app.notify("Checking for updates...")

        if not GIT_AVAILABLE:
            version_label.update(
                f"Branch: {self.git_info[0]} | Version: {self.git_info[1]} | N/A"
            )
            version_label.classes = "warning"

            log("Can't check for updates, [bold]gitpython[/bold] is not installed.")
            self.notify("You can't update because [bold]gitpython[/bold] is not installed. Use pip to install it.", title="Can't Update", severity="warning", timeout=6)
            return
        
        

        # Initialize the repository object
        repo = git.Repo(os.getcwd())

        # Fetch the latest changes from the remote (to ensure you have the latest state)
        repo.remotes.origin.fetch()

        # Get the current branch and its corresponding remote tracking branch
        current_branch = repo.active_branch
        remote_ref = f'origin/{current_branch.name}'

        # Compare the current branch with the remote tracking branch
        #commits_behind = len(list(repo.iter_commits(f'{remote_ref}..{current_branch}')))

        # how many commits the main branch is ahead of us
        commits_ahead = len(list(repo.iter_commits(f'{current_branch}..{remote_ref}')))

        # re enable the check for updates button
        self.query_one("#check-update-button").disabled = False

        updates_available = commits_ahead > 0

        if updates_available:
            version_label.update(
                f"Branch: {self.git_info[0]} | Version: {self.git_info[1]} | Out of date!"
            )
            version_label.classes = "error"
        else:
            version_label.update(
                f"Branch: {self.git_info[0]} | Version: {self.git_info[1]} | Up to date!"
            )
            version_label.classes = "success"

        return updates_available

    @work(thread=True, group="updates", name="check-updates")
    def update(self):
        log("Downloading updates from git repo..")
        self.query_one("#update-button").disabled = True

        self.notify("Applying updates...")
        repo = git.Repo(os.getcwd())
        repo.remotes.origin.fetch()

        #current_branch = repo.active_branch

        try:
            repo.remotes.origin.pull()
        except git.GitCommandError as e:
            self.notify(f"Updates failed to apply:\n{e}", title="Woops!", severity="error")
            self.query_one("#update-button").disabled = False
            return
        
        self.git_info = self.get_git_info()
        version_label = self.query_one("#version-label")

        version_label.update(
            f"Branch: {self.git_info[0]} | Version: {self.git_info[1]} | Up to date!"
        )
        version_label.classes = "success"

        self.notify("Done! Close and reopen the app for updates to complete.", title="Updates Complete")

    @work(thread=True, group="dictionary", exclusive=True)
    def update_dictionary_table(self, table, search, include_informal_words: bool = True):
        search = search.strip().lower()
        
        num_words = 0
        for gorgus, english in translation_dictionary.items():
            if gorgus.startswith("<") and gorgus.endswith(">"): # ensure we don't include internal words
                continue
            
            def search_in_string_or_list(search_query, data):
                if isinstance(data, list):
                    for string in data:
                        if search_query in string:
                            return True
                    return False
                elif isinstance(data, str):
                    return search_query in data
                else:
                    raise TypeError("Data must be a string or a list of strings")
                
            if not search_in_string_or_list(search, gorgus) and not search_in_string_or_list(search, english):
                continue

            info = []
            if gorgus in dictionary_information.get("informal_words"):
                if not include_informal_words:
                    continue

                info.append("[red]informal[/red]")
            extra_info = dictionary_information.get("extra_info").get(gorgus)
            if extra_info:
                info.append(extra_info)
            info = ', '.join(info)

            if type(english) == str:
                table.add_row(f"[blue]{gorgus}[/blue]", f"[green]{english}[/green]", info)
            elif type(english) == list:
                table.add_row(f"[blue]{gorgus}[/blue]", f"[green]{', '.join(english)}[/green]", info)
            num_words += 1

        if num_words == 0:
            table.add_row("[blue]Hmm..[/blue]", "[green]No search results found, sorry.[/green]", "[red]:([/red]")

    @work(thread=True, group="delete-settings", exclusive=True)
    def delete_settings(self):
        delete_settings_button = self.query_one("#delete-settings-button")

        if self.deleting_settings == False:
            self.notify("You cannot undo this action! Click the \"Clear Settings\" button again within 3 seconds to complete this action. Wait 3 seconds to cancel.", severity="warning", title="Watch out!", timeout=3)

            self.deleting_settings = True
            delete_settings_button.label = "Are you sure?"
            sleep(3)

            if self.deleting_settings == False:
                return

            self.notify("You didn't press the button again, your settings have not been cleared.", title="Action cancelled", severity="warning")

            self.deleting_settings = False
            delete_settings_button.label = "Clear Settings"
        else:
            self.deleting_settings = False

            if os.path.isfile("settings.json"):
                os.remove("settings.json")

            settings = get_settings()
            for widget in self.query(".setting"):
                if isinstance(widget, Checkbox):
                    widget.value = settings[widget.id]
                elif isinstance(widget, Select):
                    if widget.id == "theme-select":
                        widget.value = settings["theme_index"]

            self.notify("Settings have been cleared.", title="Done!", severity="warning")
            delete_settings_button.label = "Clear Settings"

    @on(Checkbox.Changed)
    def checkbox_changed(self, event):
        if "setting" in event.checkbox.classes:
            if event.checkbox.id in ["clock_enabled", "show_ipa"]: # certain settings require a restart to take effect
                self.notify("You need to restart for this change to take effect.", title="Setting Changed")

            if event.checkbox.id in ["add_pronounciation_accents", "formal_gorgus"]:
                self.update_translation()

            modify_json("settings.json", event.checkbox.id, event.checkbox.value)
        elif event.checkbox.id == "informal_words_checkbox":
            table = self.query_one("#dict-table")
            table.clear()
            self.update_dictionary_table(table, self.query_one("#search-input").value, event.checkbox.value) # update dictionary if the user disables informal words

    @on(Button.Pressed)
    def button_pressed(self, event):
        if event.button.id == "update-button":
            self.update()
        elif event.button.id == "check-update-button":
            self.check_for_updates()
        elif event.button.id == "delete-settings-button":
            self.delete_settings()

    @on(TextArea.Changed)
    def text_changed(self, event):
        if event.text_area.id != "translate-input": return

        self.translation_input = event.text_area.text
        self.update_translation()
    
    @on(Input.Changed)
    def search_dictionary(self, event):
        if event.input.id == "search-input":
            try:
                table: DataTable = self.query_one("#dict-table")
            except NoMatches:
                return
            
            informal_checkbox = self.query_one("#informal_words_checkbox")

            table.clear()
            self.update_dictionary_table(table, event.input.value, informal_checkbox.value)

    @on(Select.Changed)
    def select_changed(self, event):
        if event.select.id == "to-select": # change translation mode
            try:
                input_area = self.query_one("#translate-input")
            except NoMatches:
                return

            self.translation_input = self.translation
            input_area.text = self.translation_input

            self.update_translation()
        elif event.select.id == "theme-select": # changing translator theme in settings

            chosen_theme = event.select._options[event.value][0]

            modify_json("settings.json", "theme", chosen_theme)
            modify_json("settings.json", "theme_index", event.value)
            self.theme = chosen_theme
            

    @work(group="translate")
    async def update_translation(self):
        output_text_area = self.app.query_one("#output")

        settings = get_settings()
        show_ipa = settings["show_ipa"]

        if show_ipa:
            pronounciation = self.app.query_one("#pronounciation")
        translate_to_selection = self.query_one("#to-select")

        selection = translate_to_selection.value
        #self.app.notify(selection)

        should_add_accents = settings["add_pronounciation_accents"]
        pronounciation_text = ""

        if selection == 1:
            self.translation = translate(self.translation_input, "gorgus", formal=settings["formal_gorgus"], should_add_accents=should_add_accents)

            if show_ipa:
                pronounciation_text = "[dim]" + get_ipa_pronounciation(self.translation) + "[/dim]"

            #output_text_area.text = translate(text, "gorgus", should_add_accents=should_add_accents)
        elif selection == 2:
            self.translation = translate(self.translation_input, "english", should_add_accents=should_add_accents)

            #output_text_area.text = translate(text, "english", should_add_accents=should_add_accents)

        if show_ipa:
            pronounciation.update(pronounciation_text)
            output_text_area.update("[bold]" + self.translation + "[/bold]")
        else:
            output_text_area.text = self.translation

    def compose(self) -> ComposeResult:
        self.deleting_settings = False
        settings = get_settings()

        self.git_info = self.get_git_info()

        if not isinstance(self.git_info, str):
            git_version_string = f"Branch: {self.git_info[0]} | Version: {self.git_info[1]} | N/A"
        else:
            git_version_string = self.git_info

        # TODO: start using settings.get() instead of this try-except nonsense

        try:
            settings["theme_index"]
            settings["clock_enabled"]
            settings["add_pronounciation_accents"]
            settings["show_ipa"]
            settings["formal_gorgus"]
        except KeyError: # support older settings.json formats
            modify_json("settings.json", "clock_enabled", True)
            modify_json("settings.json", "theme_index", 0)
            modify_json("settings.json", "theme", "textual-dark")
            modify_json("settings.json", "add_pronounciation_accents", True)
            modify_json("settings.json", "show_ipa", True)
            modify_json("settings.json", "formal_gorgus", False)
            settings = get_settings()

        yield Header(show_clock=settings["clock_enabled"], id="header")

        with TabbedContent(id="tabs"):
            with TabPane("Translator", id="translator"): 
                yield Label("Input")
                yield TextArea(text="Hey! How are you?", tooltip="Type here!", classes="text-box", id="translate-input")
                
                yield Select([("English -> Gorgus",1), ("Gorgus -> English",2)], id="to-select", allow_blank=False,prompt="Translate...", value=1)
                yield Label("Translated")

                if settings["show_ipa"]:
                    yield CopyableLabel(id="output", copy_msg="Copied translation to keyboard!", classes="output")
                    yield CopyableLabel(id="pronounciation", copy_msg="Copied pronounciation to keyboard!", classes="output")
                    yield Label("[dim]Click on the translation or pronounciation to copy it for later![/dim]\n\n[bold]Notice:[/bold] [dim]Not a lot of words exist in Gorgus yet, so some sentences in English can't be said in Gorgus. Sorry.[/dim]", id="notice")
                else:
                    yield TextArea(classes="text-box", id="output")
                    yield Label("[bold]Notice:[/bold] [dim]Not a lot of words exist in Gorgus yet, so some sentences in English can't be said in Gorgus. Sorry.[/dim]", id="notice")

                #yield TextArea(text="Hello, how are you?", read_only=True, classes="text-box", tooltip="This is where your translated text will appear.", id="output")
            with TabPane("Dictionary",id="dict-pane"):
                yield Rule(line_style="dashed")
                with Horizontal(id="dictionary-top"):
                    with Vertical(id="lang-stats"):
                        yield Label(" [bold yellow]Language Stats[/bold yellow]")
                        yield Label(f"    - [bold]Words:[/bold] [blue]{len(translation_dictionary)}[/blue]")
                        yield Label(f"    - [bold]Phrases:[/bold] [blue]{len(phrase_translations)}[/blue]")
                    yield Input(placeholder="Search words...", id="search-input", value="")
                    yield Checkbox(label="Include informal words?", value=True, button_first=False, id="informal_words_checkbox")
                yield Rule(line_style="dashed")

                table = DataTable(id="dict-table")
                table.add_columns("Gorgus", "English", "Information")

                

                yield table
            with TabPane("Changelog"):
                try:
                    changelog_file = open("resources/CHANGELOG.md", "r")
                    changelog = changelog_file.read()
                    changelog_file.close()
                except Exception as e:
                    changelog = f"""# Woops!  
  
We failed to load the changelog, try giving **@spookydervish** on **Discord** this error message:  
{e}
                    """
                yield MarkdownViewer(changelog,open_links=False)
            with TabPane("Credits"):
                yield MarkdownViewer("""# Hello!
These are the people that make this possible! *(all of these are Discord usernames)*  
  
- **@pynecoen:** Came up with the idea to make a language and made a majority of the words
- **@spookydervish:** Made the translator, made grammar rules and made words
- **@plenorf:** Contributed many words
- **@defohumanreal:** Contributed many words, came up with the idea for the `Games` tab
- **@the-trumpeter:** Made a counting system for the language (yet to be implemented)
- **@killerpug:** Created a lot of words for adjectives / nouns
- **@projecthsi:** Made the **ENTIRE** web project and helped migrate from SpaCy!""", show_table_of_contents=False)
            with TabPane("Games"):
                with VerticalScroll() as container:
                    container.can_focus = False
                    with Center():
                        yield Markdown(GAMES_MD, id="games-md")
                    with ItemGrid(min_column_width=40, id="games-grid"):
                        for game in GAMES:
                            yield Game(game)
            with TabPane("Settings"):
                with Vertical(id="settings-panel"):
                    yield Label("Options", variant="primary", classes="settings-title")

                    yield Label("[dim]Hover on different options to see more info.[/dim]", classes="settings-note")

                    with Horizontal(classes="setting"):
                        yield Label("Check for updates when openned:")
                        yield Checkbox(button_first=False, value=True, id="check_updates_on_start", classes="setting",
                                 tooltip="When the translator is openned, if this is enabled, then it will check for available updates and notify you if they are available."
                        )

                    with Horizontal(classes="setting"):
                        yield Label("Show time in the top right:")
                        yield Checkbox(button_first=False, value=settings["clock_enabled"], id="clock_enabled", classes="setting",
                                 tooltip="Show a small clock in the top right corner of the translator."
                        )

                    with Horizontal(classes="setting"):
                        yield Label("Theme:")
                        yield Select([(theme,i) for i, theme in enumerate(self._registered_themes.keys())], allow_blank=False, id="theme-select", value=get_settings()["theme_index"], classes="setting",
                               tooltip="Choose from several different colour themes for the translator."
                        )

                    with Horizontal(classes="setting"):
                        yield Label("Add accents for pronounciation:")
                        yield Checkbox(button_first=False, value=settings["add_pronounciation_accents"], id="add_pronounciation_accents", classes="setting",
                                 tooltip="Some words may have accents on some letters to help with pronounciation."
                        )

                    with Horizontal(classes="setting"):
                        yield Label("Show IPA pronounciation:")
                        yield Checkbox(button_first=False, value=settings["show_ipa"], id="show_ipa", classes="setting",
                                 tooltip="The translator will show an IPA transcription when translating to Gorgus."
                        )

                    with Horizontal(classes="setting"):
                        yield Label("Formal Gorgus:")
                        yield Checkbox(button_first=False, value=settings["formal_gorgus"], id="formal_gorgus", classes="setting",
                                 tooltip="Gorgus is more verbose, and looks a bit more like Latin."
                        )

                    yield Label("Actions", variant="primary", classes="settings-title")
                    
                    yield Label("[dim]You can press the \"Update\" button when updates are available.[/dim]", classes="settings-note")

                    with Horizontal(id="settings-actions"):
                        yield Button("Update", variant="success", disabled=True, id="update-button", classes="setting-button", tooltip="Apply updates")
                        yield Button("Check for updates", id="check-update-button", classes="setting-button", tooltip="Check for updates")
                        yield Button("Clear Settings", variant="error", id="delete-settings-button", tooltip="Delete all settings.")

                    yield Label(
                        f"{git_version_string}",
                        variant="warning",
                        id="version-label"
                    )

        yield Footer()

    def on_worker_state_changed(self, event):
        worker = event.worker
        if worker.name == "check-updates":
            if worker.state == WorkerState.SUCCESS:
                log(f"Updates available: {worker.result}")
                self.query_one("#update-button").disabled = not worker.result

                self.query_one("#version-label").classes = not worker.result and "success" or "error"

                if worker.result == True: # There are updates available!
                    self.notify("Updates available! Go to settings to apply them.", title="Updates Available", severity="warning")
                elif worker.result == False: # Up to date
                    self.notify("No updates available, you're up to date!", title="No Updates Available")
            elif worker.state == WorkerState.ERROR:
                try:
                    self.app.query_one("#update-button").disabled = True
                    self.app.query_one("#check-update-button").disabled = False
                    self.app.query_one("#version-label").classes = "error"
                    git_version_string = f"Branch: {self.git_info[0]} | Version: {self.git_info[1]} | Failed to check for updates!"
                    self.app.query_one("#version-label").update(git_version_string)
                except NoMatches: # i hate MacOS
                    pass
                self.app.notify("Failed to check for updates. :(", severity="error", timeout=10)

    def on_mount(self):
        self.update_dictionary_table(self.query_one("#dict-pane").query_one("#dict-table"), "")

        # get the user's settings
        settings = get_settings()

        # check for updates if the user has the "check for updates on start" setting enabled
        try:
            self.query_one("#check_updates_on_start").value = settings["check_updates_on_start"]
            if settings["check_updates_on_start"]:
                self.check_for_updates()
        except KeyError:
            log("Settings failed to load due to KeyError! Maybe an update broke their settings.json?")
            self.notify(
                "Your settings may be out of date. Please delete the [bold]settings.json[/bold] in the translator's directory and restart the translator.",
                title="Settings Failed to Load",
                severity="error",
                timeout=10
            )

    def on_ready(self):
        self.push_screen(MessageBox())

        
        self.app.notify(
            message="Please be mindful, not a lot of English words exist in Gorgus, and the translator is not perfect yet.",
            title="Welcome!",
            severity="information",
            timeout=10
        )


if __name__ == "__main__":
    # clear the screen
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")

    # start the app
    app = GorgusTranslator()
    app.run()
