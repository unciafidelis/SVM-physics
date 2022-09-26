# SVM-physics

This repository contains an implementation of support vector machines with public data.


## Visual Studio Code remote conection

To make a connection with a SSH remote server you need to:

### Windows:

1. Download and install Visual Studio Code [Here](https://code.visualstudio.com/)
2. Install Open SSH [More information here](https://learn.microsoft.com/es-mx/windows-server/administration/openssh/openssh_install_firstuse)
3. Open a Power Shell terminal and put:
  ```
  ssh -Y username@servername
  ```
  and your password.
  
This step is just to prove the connection with the server.

4. Download and install Remote Develovment Vs Code extension [Here](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) 
5. Open Vs Code and look for the remote development icon in the left side tool bar, click on it, you'll be asked to install Docker also; Download and Install Docker.

6. In the remote development panel you can create a remote SSH connection, go to the panel (by clicking on the remote development icon) and look in the command prompt input, look for:

```
SSH remote connection
```
click on it.

7. You'll be asked for a SSH connection command, put:
```
ssh -Y username@servername
```
and your password.

#### Congratulations you stablished a connection between Your local VS code in Windows and the remote server with SSH!

### MacOS:

1. Download and install Visual Studio Code [Here](https://code.visualstudio.com/)
2. Open a Terminal and put:
  ```
  ssh -Y username@servername
  ```
  and your password.
  
This step is just to prove the connection with the server.

3. Download and install Remote Develovment Vs Code extension [Here](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) 
4. Open Vs Code and look for the remote development icon in the left side tool bar, click on it, you'll be asked to install Docker also; Download and Install Docker.

5. In the remote development panel you can create a remote SSH connection, go to the panel (by clicking on the remote development icon) and look in the command prompt input:

```
SSH remote connection
```
click on it.

6. You'll be asked for a SSH connection command, put:
```
ssh -Y username@servername
```
and your password.

### Congratulations you stablished a connection between Your local VS code MAcOS and the remote server!
