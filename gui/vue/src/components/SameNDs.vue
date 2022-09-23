<template>
  <v-container>
    <v-row>
      <v-col cols="12">
        <v-stepper
            v-model="e6"
            vertical
        >

          <!-- STEP 1 -->
          <v-stepper-step
              :complete="e6 > 1"
              step="1"
          >
            Input path
            <small>Select the path to the folder with your input images</small>
          </v-stepper-step>
          <v-stepper-content step="1">
            <v-file-input
                @click.prevent="selectInputFolder"
                @click:clear="fileInputValue = null; fileOutputValue = null"
                label="Input"
                :value="fileInputValue"
            />

            <v-btn
                small
                color="primary"
                @click="e6 = 2"
                :disabled="!fileInputValue"
            >
              Next
            </v-btn>
            <v-btn text small @click="$emit('back')" class="ml-3">
              Home
            </v-btn>
          </v-stepper-content>

          <!-- STEP 2 -->
          <v-stepper-step
              :complete="e6 > 2"
              step="2"
          >
            Pixel size
            <small>Enter your pixel size in microns. (You can check this, for example, in ImageJ/Fiji)</small>
          </v-stepper-step>
          <v-stepper-content step="2">
            <v-text-field
                v-model="globals.pixel"
                class="mt-0 pt-0"
                type="number"
                step="0.01"
                style="width: 60px"
            />

            <v-btn
                small
                color="primary"
                @click="setPixel"
            >
              Next
            </v-btn>
            <v-btn text small @click="e6 = 1" class="ml-3">
              Back
            </v-btn>
          </v-stepper-content>

          <!-- STEP 3 -->
          <v-stepper-step
              :complete="e6 > 3"
              step="3"
          >
            Output path
            <small>Select a path to an output folder where results will be saved.</small>
          </v-stepper-step>
          <v-stepper-content step="3">

            <v-file-input
                @click.prevent="selectOutputFolder"
                @click:clear="fileOutputValue = null"
                label="Output"
                :value="fileOutputValue"
            />
            <v-btn
                small
                color="primary"
                @click="run"
                :disabled="!fileOutputValue"
            >
              Next
            </v-btn>
            <v-btn text small @click="e6 = 2" class="ml-3">
              Back
            </v-btn>
          </v-stepper-content>

          <!-- STEP 4 -->
          <v-stepper-step step="4">
            Run
            <small>NanoNet will automatically start running</small>
          </v-stepper-step>
          <v-stepper-content step="4">
            <v-progress-linear
                rounded
                readonly
                :value="progress"
                height="15"
            >
              <strong style="color: white; font-size: 10px">{{ progressCurrentStep }} / {{ progressSteps }}</strong>
            </v-progress-linear>

            <div style="display: flex;overflow-x: auto;" class="mt-5">
              <div v-for="image in images" :key="image.title" style="margin: 0 0 0 5px">
                <v-dialog
                    v-model="image.show"
                    fullscreen
                    hide-overlay
                    transition="dialog-bottom-transition"
                >
                  <template v-slot:activator="{ on }">
                    <button v-on="on">
                      <v-card style="border-radius: 3px;width: 100px">
                        <v-img :src="srcFromImage(image.src)"/>
                      </v-card>
                    </button>

                  </template>
                  <v-card style="background-color: black">
                    <v-toolbar
                        dark
                        color="primary"
                    >
                      <v-btn
                          icon
                          dark
                          @click="image.show = false"
                      >
                        <v-icon>mdi-close</v-icon>
                      </v-btn>
                      <v-toolbar-title>
                        {{ image.title }}
                      </v-toolbar-title>
                    </v-toolbar>

                    <v-img :src="srcFromImage(image.src)" style="height: calc(100vh - 56px)"/>
                  </v-card>
                </v-dialog>
              </div>
            </div>

            <v-row>
              <v-col class="mt-5">
                <v-btn
                    small
                    color="primary"
                    @click="run"
                    :disabled="!done"
                >
                  Rerun
                </v-btn>
                <v-btn outlined small color="error" class="ml-3" @click="cancel" :disabled="done">
                  Cancel
                </v-btn>
                <v-btn text small @click="e6 = 3" class="ml-3" :disabled="!done">
                  Back
                </v-btn>
                <v-btn text small @click="$emit('back')" class="ml-3" :disabled="!done">
                  Home
                </v-btn>
              </v-col>
            </v-row>
          </v-stepper-content>
        </v-stepper>
      </v-col>
    </v-row>
  </v-container>
</template>

<script>
export default {
  name: 'SameMDs',
  data() {
    return {
      dialog: false,
      e6: 1,
      fileInputValue: null,
      fileOutputValue: null,
      progressCurrentStep: 0,
      progressSteps: 0,
      progress: 0,
      images: [],
      done: false,
      globals: {
        pixel: 9.02,
      }
    }
  },
  methods: {
    selectInputFolder() {
      if (typeof eel === 'undefined') {
        this.fileInputValue = new File(["dummy"], "dummy/path");
        return;
      }

      var that = this;
      // eslint-disable-next-line no-undef
      eel.select_path_1()(function (path) {
        that.fileInputValue = new File(["folder"], path);
      })
    },
    selectOutputFolder() {
      if (typeof eel === 'undefined') {
        this.fileOutputValue = new File(["dummy"], "dummy/path");
        return;
      }

      var that = this;
      // eslint-disable-next-line no-undef
      eel.select_output_path()(function (path) {
        that.fileOutputValue = new File(["folder"], path);
      })
    },
    setPixel() {
      this.e6 = 3;
      if (typeof eel === 'undefined') {
        return;
      }

      // eslint-disable-next-line no-undef
      eel.set_pixel(this.globals.pixel);
    },
    cancel() {
      if (typeof eel === 'undefined' || this.progressCurrentStep === this.progressSteps) {
        return;
      }

      this.done = true;

      // eslint-disable-next-line no-undef
      eel.cancel_same_nds();
    },
    run() {
      this.done = false;
      this.progress = 0;
      this.progressCurrentStep = 0;
      this.images = [];
      if (this.e6 === 3) {
        this.e6 = 4;
      }

      if (typeof eel === 'undefined') {
        return;
      }

      const that = this;

      // eslint-disable-next-line no-undef
      eel.run_same_nds()(function () {
        that.done = true;
      });
    },
    srcFromImage(image) {
      return 'data:image/png;base64,' + image
    }
  },
  mounted() {
    if (typeof eel === 'undefined') {
      return;
    }

    function addImageUnbound(image, p, i) {
      image.show = false;
      this.progress = (100 / p) * i;
      this.progressCurrentStep = i;
      this.progressSteps = p;

      return this.images.push(image);
    }

    var addImageSameNDs = addImageUnbound.bind(this);

    // eslint-disable-next-line no-undef
    eel.expose(addImageSameNDs, 'add_image_same_nd');
  }
}
</script>
