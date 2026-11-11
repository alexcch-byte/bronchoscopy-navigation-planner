' Starts the bronchoscopy planner test console (http://127.0.0.1:8000) hidden.
' Meant to live in the Windows Startup folder (shell:startup) so the console and its Tailscale URL
' come back after a reboot. Waits for the Google Drive G: drive, which may not be mounted yet at login.
' Set BRONCHNAV_SERVER_DIR to the full path of the repo's server folder (default: the folder this script is in).
' Log: %USERPROFILE%\bronchnav-console.log. Delete the copy in the Startup folder to stop it starting.
Set sh  = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
serverDir = sh.ExpandEnvironmentStrings("%BRONCHNAV_SERVER_DIR%")
If Left(serverDir, 1) = "%" Then serverDir = fso.GetParentFolderName(WScript.ScriptFullName)
py = sh.ExpandEnvironmentStrings("%USERPROFILE%") & "\venvs\bronchnav\Scripts\python.exe"
log = sh.ExpandEnvironmentStrings("%USERPROFILE%") & "\bronchnav-console.log"
tries = 0
Do While (Not fso.FolderExists(serverDir)) And tries < 150
  WScript.Sleep 2000
  tries = tries + 1
Loop
If fso.FolderExists(serverDir) And fso.FileExists(py) Then
  sh.CurrentDirectory = serverDir
  sh.Run "cmd /c """"" & py & """ run_local.py >> """ & log & """ 2>&1""", 0, False
End If
