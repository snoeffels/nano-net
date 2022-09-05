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
            Select input path
            <small>Summarize if needed</small>
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
            Type in pixel size
            <small>Summarize if needed</small>
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
            Observe segmentation
            <small>Summarize if needed</small>
          </v-stepper-step>
          <v-stepper-content step="3">
            <v-checkbox
                class="ml-2"
                :input-value="globals.display"
                @change="globals.display = !globals.display; setDisplayResults()"
                label="Display segmentations while processing"
            />

            <!--            <v-tooltip bottom>-->
            <!--              <template #activator="{ on }">-->
            <!--                <v-icon color="red" class="mr-1" v-on="on">fab fa-youtube</v-icon>-->
            <!--              </template>-->
            <!--              <span><p>-->
            <!--              image: explanation...<br>-->
            <!--              thresh3: explanation...<br>-->
            <!--              closed3: explanation...<br>-->
            <!--              cleared3: explanation...<br>-->
            <!--              rect_w: explanation...<br>-->
            <!--            </p></span>-->
            <!--            </v-tooltip>-->

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
            Save results
            <small>Summarize if needed</small>
          </v-stepper-step>
          <v-stepper-content step="4">
            <h3>Done!</h3>
            <p>
              Thanks for using....
              Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore
              et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum.
              Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.
            </p>
            <v-btn
                small
                color="primary"
                @click="console.log('rerun')"
            >
              Rerun
            </v-btn>
            <v-btn text small @click="e6 = 3" class="ml-3">
              Back
            </v-btn>
            <v-btn text small @click="$emit('back')" class="ml-3">
              Home
            </v-btn>
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
      e6: 1,
      fileInputValue: null,
      fileOutputValue: null,
      globals: {
        pixel: 9.02,
        display: false
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
      eel.select_ms_path()(function (path) {
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
    setDisplayResults() {
      if (typeof eel === 'undefined') {
        return;
      }

      // eslint-disable-next-line no-undef
      eel.set_display_results(this.globals.display);
    },
    run() {
      if (this.e6 === 3) {
        this.e6 = 4;
      }

      if (typeof eel === 'undefined') {
        return;
      }

      // eslint-disable-next-line no-undef
      eel.run_same_nds();
    }
  },
}
</script>
