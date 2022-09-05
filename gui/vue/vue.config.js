const { defineConfig } = require('@vue/cli-service');
require('path');
module.exports = defineConfig({
  devServer: {
    port: 10000,
  },
  transpileDependencies: [
    'vuetify'
  ],
  outputDir: "../dist"
})
