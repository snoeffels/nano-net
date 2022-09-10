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
            Select inputs
            <small>Summarize if needed</small>
          </v-stepper-step>
          <v-stepper-content step="1">
            <p>All whitespace or non alphanumerical characters will be removed from labels. Please specify at least two
              inputs with unique names to continue.</p>

            <v-row>
              <v-col>
                <fieldset
                    style="border: 1px rgba(113,113,113,0.4) solid; border-radius: 5px; padding-left: 10px; padding-right: 20px">

                  <legend style="margin-left: 24px; padding: 0.2em 0.8em; color: rgba(0,0,0,0.6);font-size: 12px">Input
                    1
                  </legend>
                  <div style="display:flex">
                    <v-text-field
                        v-model="conditionName"
                        @change="handleNameChange"
                        label="Name"
                        style="padding-left: 33px; padding-top: 20px; width: 80%"
                    />

                    <input
                        v-model="color"
                        type="color"
                        style="width: 33px;margin-top: 34px;padding-left: 10px"
                    />
                  </div>

                  <v-file-input
                      @click.prevent="selectInputFolder"
                      @click:clear="fileInputValue = null;"
                      label="Input"
                      :value="fileInputValue"
                      style="padding-top: 0;margin-top: -3px"
                  />

                </fieldset>
              </v-col>
              <v-col>
                <fieldset
                    style="border: 1px rgba(113,113,113,0.4) solid; border-radius: 5px; padding-left: 10px; padding-right: 20px">

                  <legend style="margin-left: 24px; padding: 0.2em 0.8em; color: rgba(0,0,0,0.6);font-size: 12px">Input
                    2
                  </legend>
                  <div style="display:flex">
                    <v-text-field
                        v-model="condition2Name"
                        @change="handleNameChange2"
                        label="Name"
                        style="padding-left: 33px; padding-top: 20px; width: 80%"
                    />

                    <input
                        v-model="color2"
                        type="color"
                        style="width: 33px;margin-top: 34px;padding-left: 10px"
                    />
                  </div>


                  <v-file-input
                      @click.prevent="selectInputFolder2"
                      @click:clear="fileInput2Value = null;"
                      label="Input"
                      :value="fileInput2Value"
                      style="padding-top: 0;margin-top: -3px"
                  />
                </fieldset>

              </v-col>
            </v-row>
            <v-row class="pb-5">
              <v-col>
                <fieldset
                    style="border: 1px rgba(113,113,113,0.4) solid; border-radius: 5px; padding-left: 10px; padding-right: 20px">

                  <legend style="margin-left: 24px; padding: 0.2em 0.8em; color: rgba(0,0,0,0.6);font-size: 12px">Input
                    3
                  </legend>
                  <div style="display:flex">
                    <v-text-field
                        v-model="condition3Name"
                      @change="handleNameChange3"
                        label="Name"
                        style="padding-left: 33px; padding-top: 20px; width: 80%"
                    />

                    <input
                        v-model="color3"
                        type="color"
                        style="width: 33px;margin-top: 34px;padding-left: 10px"
                    />
                  </div>

                  <v-file-input
                      @click.prevent="selectInputFolder3"
                      @click:clear="fileInput3Value = null;"
                      label="Input"
                      :value="fileInput3Value"
                      style="padding-top: 0;margin-top: -3px"
                  />
                </fieldset>

              </v-col>
              <v-col>
                <fieldset
                    style="border: 1px rgba(113,113,113,0.4) solid; border-radius: 5px; padding-left: 10px; padding-right: 20px">

                  <legend style="margin-left: 24px; padding: 0.2em 0.8em; color: rgba(0,0,0,0.6);font-size: 12px">Input
                    4
                  </legend>
                  <div style="display:flex">
                    <v-text-field
                        v-model="condition4Name"
                      @change="handleNameChange4"
                        label="Name"
                        style="padding-left: 33px; padding-top: 20px; width: 80%"
                    />

                    <input
                        v-model="color4"
                        type="color"
                        style="width: 33px;margin-top: 34px;padding-left: 10px"
                    />
                  </div>

                  <v-file-input
                      @click.prevent="selectInputFolder4"
                      @click:clear="fileInput4Value = null;"
                      label="Input"
                      :value="fileInput4Value"
                      style="padding-top: 0;margin-top: -3px"
                  />
                </fieldset>


              </v-col>
            </v-row>


            <v-btn
                small
                color="primary"
                @click="e6 = 2"
                :disabled="[fileInputValue, fileInput2Value, fileInput3Value, fileInput4Value].filter(e => e !== null).length < 2"
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
            Select input path NaCl
            <small>Summarize if needed</small>
          </v-stepper-step>
          <v-stepper-content step="2">
            <v-file-input
                @click.prevent="selectInputFolder2"
                @click:clear="fileInput2Value = null"
                label="Input"
                :value="fileInput2Value"
            />

            <v-btn
                small
                color="primary"
                @click="e6 = 3"
                :disabled="!fileInput2Value"
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
            Order of conditions
            <small>Summarize if needed</small>
          </v-stepper-step>
          <v-stepper-content step="3">
            <v-row
                align="center"
            >
              <v-col cols="12">
                <v-autocomplete
                    v-model="globals.order"
                    :items="orderItems"
                    dense
                    chips
                    small-chips
                    multiple
                    style="width: 280px"
                />
              </v-col>
            </v-row>


            <v-btn
                small
                color="primary"
                @click="e6 = 4"
                :disabled="false"
            >
              Next
            </v-btn>
            <v-btn text small @click="e6 = 2" class="ml-3">
              Back
            </v-btn>
          </v-stepper-content>

          <!-- STEP 4 -->
          <v-stepper-step
              :complete="e6 > 4"
              step="4"
          >
            Parameter settings
            <small>Summarize if needed</small>
          </v-stepper-step>
          <v-stepper-content step="4">
            <v-row>
              <v-col cols="6">
                <h4>
                  Random Forest
                </h4>
                <v-row>
                  <v-col>
                    <v-text-field
                        v-model="globals.testSize"
                        label="test_size"
                        type="number"
                        step="0.01"
                    />
                    <v-text-field
                        v-model="globals.nEstimators"
                        label="N-Estimators"
                        type="number"
                        step="1"
                    />
                    <v-text-field
                        v-model="globals.minSamplesSplit"
                        label="Min samples split"
                        type="number"
                        step="1"
                    />
                  </v-col>
                  <v-col>
                    <v-text-field
                        v-model="globals.minSamplesLeaf"
                        label="Min samples leaf"
                        type="number"
                        step="1"
                    />
                    <v-text-field
                        v-model="globals.maxFeatures"
                        label="Max features"
                    />
                    <v-text-field
                        v-model="globals.maxDepth"
                        label="Max depth"
                        type="number"
                        step="1"
                    />
                  </v-col>
                </v-row>
              </v-col>
              <v-col cols="3">
                <h4>
                  TSNE
                </h4>
                <v-text-field
                    v-model="globals.nComponents"
                    label="N-Components"
                    type="number"
                    step="0.01"
                />
                <v-text-field
                    v-model="globals.nNeighbors"
                    label="Perplexity"
                    type="number"
                    step="1"
                />
                <v-text-field
                    v-model="globals.nIter"
                    label="N-Iterations"
                    type="number"
                    step="1"
                />
              </v-col>
              <v-col cols="3">
                <h4>
                  K-NN
                </h4>
                <v-text-field
                    v-model="globals.testSizeKnn"
                    label="test_size"
                    type="number"
                    step="0.01"
                />
                <v-text-field
                    v-model="globals.nNeighbors"
                    label="N-Estimators"
                    type="number"
                    step="1"
                />
              </v-col>
            </v-row>
            <v-btn
                small
                color="primary"
                @click="e6 = 5"
                :disabled="false"
            >
              Next
            </v-btn>
            <v-btn text small @click="e6 = 3" class="ml-3">
              Back
            </v-btn>
          </v-stepper-content>

          <!-- STEP 5 -->
          <v-stepper-step
              :complete="e6 > 5"
              step="5"
          >
            Estimation of K-NN optimal number
            <small>Summarize if needed</small>
          </v-stepper-step>
          <v-stepper-content step="5">
            <v-text-field
                v-model="globals.acuracyKnn"
                label="K-NN Acuracy"
                type="number"
                step="0.01"
                readonly
                style="width: 150px"
            />
            <v-text-field
                v-model="globals.optimalNumberOfNeighbors"
                label="Optimal number of neighbors"
                type="number"
                step="1"
                style="width: 150px"
                readonly
            />

            <v-btn
                small
                color="primary"
                @click="e6 = 6"
                :disabled="false"
            >
              Next
            </v-btn>
            <v-btn text small @click="e6 = 4" class="ml-3">
              Back
            </v-btn>
          </v-stepper-content>

          <!-- STEP 6 -->
          <v-stepper-step
              :complete="e6 > 6"
              step="6"
          >
            Observe segmentation
            <small>Summarize if needed</small>
          </v-stepper-step>
          <v-stepper-content step="6">
            <v-btn
                small
                color="primary"
                @click="e6 = 7"
                :disabled="false"
            >
              Next
            </v-btn>
            <v-btn text small @click="e6 = 5" class="ml-3">
              Back
            </v-btn>
          </v-stepper-content>

          <!-- STEP 7 -->
          <v-stepper-step
              :complete="e6 > 7"
              step="7"
          >
            Observe segmentation
            <small>Summarize if needed</small>
          </v-stepper-step>
          <v-stepper-content step="7">
            <v-checkbox
                class="ml-2"
                :input-value="globals.display"
                @change="globals.display = !globals.display; setDisplayResults()"
                label="Display segmentations while processing"
            />

            <v-file-input
                @click.prevent="selectOutputFolder"
                @click:clear="fileOutputValue = null"
                label="Output"
                :value="fileOutputValue"
            />
            <v-btn
                small
                color="primary"
                @click="e6 = 8"
                :disabled="!fileOutputValue"
            >
              Next
            </v-btn>
            <v-btn text small @click="e6 = 6" class="ml-3">
              Back
            </v-btn>
          </v-stepper-content>

          <!-- STEP 8 -->
          <v-stepper-step step="8">
            Save results
            <small>Summarize if needed</small>
          </v-stepper-step>
          <v-stepper-content step="8">
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
            <v-btn text small @click="e6 = 7" class="ml-3">
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
  name: 'DifferentNDs',
  data() {
    return {
      e6: 1,
      fileInputValue: null,
      color: '#fd8f00',
      conditionName: "Label1",
      fileInput2Value: null,
      color2: '#0063b9',
      condition2Name: "Label2",
      fileInput3Value: null,
      color3: '#4b9d00',
      condition3Name: "Label3",
      fileInput4Value: null,
      color4: '#da00d7',
      condition4Name: "Label4",
      fileOutputValue: null,
      orderItems: ['MS', 'NaCl', 'Sorbitol'],
      value: null,
      globals: {
        order: [''],
        testSize: 0.2,
        nEstimators: 1000,
        minSamplesSplit: 2,
        minSamplesLeaf: 1,
        maxFeatures: 'sqrt',
        maxDepth: 110,
        testSizeKnn: 0.2,
        nNeighbors: 3,
        nComponents: 2,
        perplexity: 20,
        nIter: 500,
        acuracyKnn: 0.75,
        optimalNumberOfNeighbors: 5,
        display: false
      }
    }
  },
  methods: {
    abc() {
      console.log('abc');
    },
    handleNameChange() {
      this.conditionName = this.conditionName.replace(/[^a-zA-Z0-9]+/gi, '');
      if (this.conditionName === '' || this.conditionName === this.condition2Name || this.conditionName === this.condition3Name || this.conditionName === this.condition4Name)
        this.conditionName = 'Label1';
    },
    handleNameChange2() {
      this.condition2Name = this.condition2Name.replace(/[^a-zA-Z0-9]+/gi, '');
      if (this.condition2Name === '' || this.condition2Name === this.conditionName || this.condition2Name === this.condition3Name || this.condition2Name === this.condition4Name)
        this.condition2Name = 'Label2';
    },
    handleNameChange3() {
      this.condition3Name = this.condition3Name.replace(/[^a-zA-Z0-9]+/gi, '');
      if (this.condition3Name === '' || this.condition3Name === this.conditionName || this.condition3Name === this.condition2Name || this.conditionName === this.condition4Name)
        this.condition3Name = 'Label3';
    },
    handleNameChange4() {
      this.condition4Name = this.condition4Name.replace(/[^a-zA-Z0-9]+/gi, '');
      if (this.condition4Name === '' || this.condition4Name === this.condition2Name || this.condition4Name === this.condition3Name || this.condition4Name === this.conditionName)
        this.condition4Name = 'Label4';
    },
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
    selectInputFolder2() {
      if (typeof eel === 'undefined') {
        this.fileInput2Value = new File(["dummy"], "dummy/path");
        return;
      }

      var that = this;
      // eslint-disable-next-line no-undef
      eel.select_nacl_path()(function (path) {
        that.fileInput2Value = new File(["folder"], path);
      })
    },
    selectInputFolder3() {
      if (typeof eel === 'undefined') {
        this.fileInput3Value = new File(["dummy"], "dummy/path");
        return;
      }

      var that = this;
      // eslint-disable-next-line no-undef
      eel.select_nacl_path()(function (path) {
        that.fileInput3Value = new File(["folder"], path);
      })
    },
    selectInputFolder4() {
      if (typeof eel === 'undefined') {
        this.fileInput4Value = new File(["dummy"], "dummy/path");
        return;
      }

      var that = this;
      // eslint-disable-next-line no-undef
      eel.select_nacl_path()(function (path) {
        that.fileInput4Value = new File(["folder"], path);
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
