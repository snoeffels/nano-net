<template>
  <div class="hello">
    <h1>{{ this.message }}</h1>
    <button @click="changeText">change text</button>
    <input type="text" v-model="inputValue"/>
    <button @click="onClick">Send</button>
    <div>
      {{ response }}
    </div>
  </div>
</template>

<!-- TODO: find a way to avoid these -->
<!-- Import this on every component using python implementations  -->
<script type="text/javascript" src="/eel.js"></script>

<script>
export default {
  name: 'HelloWorld',
  data: function () {
    return {
      message: "",
      inputValue: "",
      response: ""
    }
  },
  mounted: function () {
    eel.hello_world()((val) => {
      // Receiving a value from Python
      this.message = val;
    })
  },
  methods: {
    changeText() {
      this.message = "hallo was geht?"
    },
    onClick() {
      // Passing values to Python
      eel.print_string(this.inputValue)((val) => {
        // Return response from Python
        this.response = val
      })
    }
  }
}
</script>