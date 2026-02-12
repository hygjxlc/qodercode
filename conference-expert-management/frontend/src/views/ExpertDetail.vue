<template>
  <div class="expert-detail">
    <el-page-header @back="goBack" title="专家详情" />
    
    <el-card class="detail-card" v-loading="loading">
      <template #header>
        <div class="card-header">
          <span>{{ expert.name }}</span>
          <el-tag :type="getStatusType(expert.contactStatus)">{{ expert.contactStatus }}</el-tag>
        </div>
      </template>
      
      <el-descriptions :column="2" border>
        <el-descriptions-item label="邮箱">{{ expert.email }}</el-descriptions-item>
        <el-descriptions-item label="电话">{{ expert.phone }}</el-descriptions-item>
        <el-descriptions-item label="专业领域">{{ expert.professionalField }}</el-descriptions-item>
        <el-descriptions-item label="职称">{{ expert.title }}</el-descriptions-item>
        <el-descriptions-item label="所属机构">{{ expert.organization }}</el-descriptions-item>
        <el-descriptions-item label="地址">{{ expert.address }}</el-descriptions-item>
        <el-descriptions-item label="创建时间">{{ formatDate(expert.createdAt) }}</el-descriptions-item>
        <el-descriptions-item label="更新时间">{{ formatDate(expert.updatedAt) }}</el-descriptions-item>
      </el-descriptions>
      
      <div class="actions">
        <el-button type="primary" @click="viewContactRecords">
          <el-icon><ChatLineRound /></el-icon>查看联系记录
        </el-button>
        <el-button type="success" @click="viewConferenceRecords">
          <el-icon><Calendar /></el-icon>查看会务历史
        </el-button>
        <el-button @click="editExpert">编辑</el-button>
      </div>
    </el-card>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { expertApi } from '../api/expert'

export default {
  name: 'ExpertDetail',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const loading = ref(false)
    const expert = ref({})

    const fetchExpert = async () => {
      loading.value = true
      try {
        const res = await expertApi.getExpertById(route.params.id)
        if (res.data.success) {
          expert.value = res.data.data
        }
      } catch (error) {
        ElMessage.error('获取专家信息失败')
      } finally {
        loading.value = false
      }
    }

    const goBack = () => {
      router.push('/')
    }

    const viewContactRecords = () => {
      router.push(`/expert/${route.params.id}/contacts`)
    }

    const viewConferenceRecords = () => {
      router.push(`/expert/${route.params.id}/conferences`)
    }

    const editExpert = () => {
      router.push(`/?edit=${route.params.id}`)
    }

    const getStatusType = (status) => {
      const types = {
        '待联系': 'info',
        '已联系': 'warning',
        '确认参加': 'success',
        '暂不参与': 'danger'
      }
      return types[status] || 'info'
    }

    const formatDate = (date) => {
      if (!date) return '-'
      return new Date(date).toLocaleString('zh-CN')
    }

    onMounted(fetchExpert)

    return {
      loading,
      expert,
      goBack,
      viewContactRecords,
      viewConferenceRecords,
      editExpert,
      getStatusType,
      formatDate
    }
  }
}
</script>

<style scoped>
.expert-detail {
  max-width: 1000px;
  margin: 0 auto;
}

.detail-card {
  margin-top: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.actions {
  margin-top: 20px;
  display: flex;
  gap: 10px;
}
</style>
