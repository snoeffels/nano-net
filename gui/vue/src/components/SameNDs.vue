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
                @click.prevent="pickFolder"
                @click:clear="fileInputValue = null"
                label="File input"
                :value="fileInputValue"
            ></v-file-input>

            <v-btn
                color="primary"
                @click="e6 = 2"
                :disabled="!fileInputValue"
            >
              Next
            </v-btn>
            <v-btn outlined @click="$emit('back')" class="ml-3">
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
            <v-card
                color="grey lighten-1"
                class="mb-12"
                height="200px"
            ></v-card>
            <v-btn
                color="primary"
                @click="e6 = 3"
            >
              Next
            </v-btn>
            <v-btn text @click="e6 = 1">
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
            <v-card
                color="grey lighten-1"
                class="mb-12"
                height="200px"
            ></v-card>
            <v-btn
                color="primary"
                @click="e6 = 4"
            >
              Next
            </v-btn>
            <v-btn text @click="e6 = 2">
              Back
            </v-btn>
          </v-stepper-content>

          <!-- STEP 4 -->
          <v-stepper-step step="4">
            Save results
            <small>Summarize if needed</small>
          </v-stepper-step>
          <v-stepper-content step="4">
            <v-card
                color="grey lighten-1"
                class="mb-12"
                height="200px"
            ></v-card>
            <v-btn
                color="primary"
                @click="e6 = 1"
            >
              Done
            </v-btn>
            <v-btn text @click="e6 = 3">
              Back
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
      fileInputValue: null
    }
  },
  methods: {
    pickFolder() {
      if (typeof eel === 'undefined') {
        this.fileInputValue = new File(["dummy"], "dummy/path");
        return;
      }

      var that = this;
      // eslint-disable-next-line no-undef
      eel.select_folder()(function (path) {
        that.fileInputValue = new File(["folder"], path);
      })
    }
  },
}
</script>
