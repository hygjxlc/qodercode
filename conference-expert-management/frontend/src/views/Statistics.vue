<template>
  <div class="statistics">
    <el-card>
      <template #header>
        <div class="card-header">
          <span>数据统计</span>
        </div>
      </template>
      
      <el-row :gutter="20">
        <el-col :span="8">
          <el-statistic title="专家总数" :value="overview.totalExperts" />
        </el-col>
      </el-row>
      
      <el-divider />
      
      <el-row :gutter="20">
        <el-col :span="12">
          <h3>联系状态分布</h3>
          <div ref="statusChart" style="height: 300px"></div>
        </el-col>
        <el-col :span="12">
          <h3>专业领域分布</h3>
          <div ref="fieldChart" style="height: 300px"></div>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<script>
import { ref, onMounted, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import api from '../api/expert'

export default {
  name: 'Statistics',
  setup() {
    const overview = reactive({
      totalExperts: 0,
      contactStatusDistribution: {},
      fieldDistribution: {}
    })
    const statusChart = ref(null)
    const fieldChart = ref(null)
    
    const fetchStatistics = async () => {
      try {
        const res = await api.get('/statistics/overview')
        if (res.data.success) {
          Object.assign(overview, res.data.data)
          initCharts()
        }
      } catch (error) {
        ElMessage.error('获取统计数据失败')
      }
    }
    
    const initCharts = () => {
      // Status chart
      if (statusChart.value) {
        const chart = echarts.init(statusChart.value)
        const statusData = Object.entries(overview.contactStatusDistribution)
          .map(([name, value]) => ({ name, value }))
        
        chart.setOption({
          tooltip: { trigger: 'item' },
          legend: { bottom: '5%' },
          series: [{
            type: 'pie',
            radius: ['40%', '70%'],
            avoidLabelOverlap: false,
            itemStyle: {
              borderRadius: 10,
              borderColor: '#fff',
              borderWidth: 2
            },
            label: { show: false },
            emphasis: {
              label: {
                show: true,
                fontSize: 20,
                fontWeight: 'bold'
              }
            },
            data: statusData
          }]
        })
      }
      
      // Field chart
      if (fieldChart.value) {
        const chart = echarts.init(fieldChart.value)
        const fieldData = Object.entries(overview.fieldDistribution)
          .map(([name, value]) => ({ name, value }))
        
        chart.setOption({
          tooltip: { trigger: 'axis' },
          xAxis: { type: 'category', data: fieldData.map(d => d.name) },
          yAxis: { type: 'value' },
          series: [{
            data: fieldData.map(d => d.value),
            type: 'bar',
            itemStyle: { color: '#409EFF' }
          }]
        })
      }
    }
    
    onMounted(fetchStatistics)
    
    return {
      overview,
      statusChart,
      fieldChart
    }
  }
}
</script>

<style scoped>
.statistics {
  max-width: 1200px;
  margin: 0 auto;
}

.card-header {
  font-size: 18px;
  font-weight: bold;
}

h3 {
  margin-bottom: 20px;
  color: #606266;
}
</style>
