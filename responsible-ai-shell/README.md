# RAI Shell

This project was generated with [Angular CLI](https://github.com/angular/angular-cli) version 15.2.9.

## Development

Before you can build this project, you must install and configure the following dependencies on your machine:

1. [Node.js][]: We use Node to run a development web server and build the project.
   Depending on your system, you can install Node either from source or as a pre-packaged bundle.
   ( preferably v18 & above)
2. [Angular CLI][]: Install Angular Cli using [npm install -g @angular/cli@15.2.9].
3. Set Environmental Variable , Path as Variable name & Value will be path to the npm. Example 
    C:\Users\<username>\AppData\Roaming\npm.
4. Now CLI will be available .

## To Run Locally

After installing Node & following above steps, you should be able to run the following command to install development tools.
You will only need to run this command when dependencies change in [package.json](package.json).

Go to the root folder of the repository & run ,

```
npm install
```
We use npm scripts and [Webpack][] as our build system.

Run the following commands in two separate terminals to create a blissful development experience where your browser
auto-refreshes when files change on your hard drive.


In start.js file uncomment await runLocally() & comment out setDataFromENV()
Then do 
```
npm start
``` 
Application is available in localhost:30010 .

Npm is also used to manage CSS and JavaScript dependencies used in this application. You can upgrade dependencies by
specifying a newer version in [package.json](package.json). You can also run `npm update` and `npm install` to manage dependencies.
Add the `help` flag on any command to see how you can use it. For example, `npm help update`.

The `npm run` command will list all of the scripts available to run for this project.

### Using Angular CLI

You can also use [Angular CLI][] to generate some custom client code.

For example, the following command:

```
ng generate component my-component
```

will generate few files:

```
create src/main/webapp/app/my-component/my-component.component.html
create src/main/webapp/app/my-component/my-component.component.ts
update src/main/webapp/app/app.module.ts
```

## Building for production

### Packaging as jar

To build the final jar and optimize the IDF application for production, run:

```


```

This will concatenate and minify the client CSS and JavaScript files. It will also modify `index.html` so it references these new files.
To ensure everything worked, run:

```


```

Then navigate to `http://localhost:` in your browser.

### Packaging as war

To package your application as a war in order to deploy it to an application server, run:

```


```


## Code scaffolding

Run `ng generate component component-name` to generate a new component. You can also use `ng generate directive|pipe|service|class|guard|interface|enum|module`.

## Build

Run `ng build` to build the project. The build artifacts will be stored in the `dist/` directory.

## Running unit tests

Run `ng test` to execute the unit tests via [Karma](https://karma-runner.github.io).

## Running end-to-end tests

Run `ng e2e` to execute the end-to-end tests via a platform of your choice. To use this command, you need to first add a package that implements end-to-end testing capabilities.

## Further help

To get more help on the Angular CLI use `ng help` or go check out the [Angular CLI Overview and Command Reference](https://angular.io/cli) page.


## Dependency
 As these are frontend modules , they have dependency on these modules
 1. responsible-ai-backend
2.	responsible-ai-admin
