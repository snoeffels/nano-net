<template>
  <div>
    <div v-if="!sameNDs && !differentNDs">
      <v-container>
        <v-row>
          <v-col cols="12">
            <h2>Home</h2>
            <p class="text-sm-body-2">
              NanoNet prompts you to choose between two main workflows: segmentation and feature extraction for one condition (A) or machine learning aided classification and feature comparison between different conditions (B). 
            </p>
            <v-row>
              <v-col cols="6">
                <h3>A: Segmentation and feature extraction</h3>
                <p class="text-sm-body-2">
                  This part of the NanoNet expects one folder with TIFF images, displaying your regions of interest (ROIs). 
                  For certain features (such as Nanodomain density) it is important that your ROIs all have the same size. 
                  The app will prompt you to enter the size in pixels. This can be checked in common image analysis tool such as Fiji. 
                  As output you will get a results excel sheet with all the features extracted. For a detailed description of feature annotation see (NanoNet Guide).
                </p>
                <v-btn small color="primary" @click="sameNDs=true">
                  Start
                </v-btn>
              </v-col>

              <v-col cols="6">
                <h3>B: Comparison and Classification between conditions</h3>
                <p class="text-sm-body-2">
                  This part of the NanoNet expects at minimum 2 input folders with TIFF images, displaying your regions of interest (ROIs) you want to compare. 
                  For certain features (such as Nanodomain density) it is important that your ROIs all have the same size. 
                  The app will prompt you to enter the size in pixels. This can be checked in common image analysis tool such as Fiji. 
                  The extracted features of each condition (image folder) will be passed on to train a Random Forest (RF) and a k-NN classifier (k-NN). 
                  Additionally you will get boxplots comparing chosen features between your conditions, a t-SNE map and an inter-feature correlation map. For a detailed description of feature annotation see (NanoNet Guide).
                </p>
                <v-btn small color="primary" @click="differentNDs=true">
                  Start
                </v-btn>
              </v-col>
            </v-row>
          </v-col>
        </v-row>
      </v-container>
    </div>
    <div>
      <SameNDs v-if="sameNDs" @back="sameNDs=false" class="mt-5"/>
      <DifferentNDs v-if="differentNDs" @back="differentNDs=false" class="mt-5"/>
    </div>
  </div>
</template>

<script>
import SameNDs from '../components/SameNDs'
import DifferentNDs from '../components/DifferentNDs'

export default {
  name: 'HomeView',
  data: () => {
    return {
      sameNDs: false,
      differentNDs: false,
    }
  },
  components: {
    SameNDs,
    DifferentNDs,
  },
}
</script>
