const { defineConfig } = require('@vue/cli-service');
require('path');
module.exports = defineConfig({
  transpileDependencies: true,
  outputDir: "../dist"
})